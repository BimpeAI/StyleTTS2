import os
import os.path as osp
import shutil
import time
import random
import warnings

import click
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from munch import Munch
from torch.utils.tensorboard import SummaryWriter

import logging
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger

warnings.simplefilter('ignore')

from models import *
from meldataset import build_dataloader
from utils import *
from losses import *
from optimizers import build_optimizer

logger = get_logger(__name__, log_level="DEBUG")


@click.command()
@click.option('-p', '--config_path', default='Configs/config_bimpe.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = accelerator.device

    epochs = config.get('epochs_1st', config.get('epochs', 200))
    log_interval = config.get('log_interval', 10)
    saving_epoch = int(config.get('save_freq', 10))
    early_stopping_patience = int(config.get('early_stopping_patience', 2))
    early_stopping_min_delta = float(config.get('early_stopping_min_delta', 0.0))
    first_stage_name = config.get('first_stage_path', 'first_stage.pth')
    metrics_csv = osp.join(log_dir, 'metrics_1st.csv')
    num_workers = int(config.get('num_workers', 0))

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    if not osp.isfile(OOD_data):
        raise FileNotFoundError(
            f"OOD text file not found: {OOD_data}. "
            "Copy Data/OOD_texts.txt from the StyleTTS2 repo into the config path, "
            "or run: python scripts/prepare_bimpe_data.py --ensure_ood"
        )

    max_len = config.get('max_len', 400)

    train_list, val_list = get_data_path_list(train_path, val_path)
    if accelerator.is_main_process:
        print(f"Train samples: {len(train_list)}, Val samples: {len(val_list)}", flush=True)
        print(f"root_path={root_path}, OOD_data={OOD_data}, num_workers={num_workers}", flush=True)
        print(f"TMA_epoch={config['loss_params'].get('TMA_epoch')}, max_len={max_len}", flush=True)

    train_dataloader = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_config={},
        device=device,
    )

    val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        device=device,
        dataset_config={},
    )

    with accelerator.main_process_first():
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        from Utils.PLBERT.util import load_plbert
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    best_loss = float('inf')
    best_epoch = -1
    patience = 0
    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch

    for k in model:
        model[k] = accelerator.prepare(model[k])

    train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)

    _ = [model[key].to(device) for key in model]

    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict={key: scheduler_params.copy() for key in model},
        lr=float(config['optimizer_params'].get('lr', 1e-4)),
    )

    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(
                model,
                optimizer,
                config['pretrained_model'],
                load_only_params=config.get('load_only_params', True),
            )
        else:
            start_epoch = 0
            iters = 0

    try:
        n_down = model.text_aligner.module.n_down
    except Exception:
        n_down = model.text_aligner.n_down

    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)

    if accelerator.is_main_process:
        print("Starting training loop...", flush=True)

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()
        epoch_mel = 0.0
        epoch_gen = 0.0
        epoch_d = 0.0
        epoch_mono = 0.0
        epoch_s2s = 0.0
        epoch_slm = 0.0
        epoch_steps = 0

        _ = [model[key].train() for key in model]

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, _, _, mels, mel_input_length, _ = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (
                    (~mask)
                    .unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = attn_mask.float() * (
                    (~text_mask)
                    .unsqueeze(-1)
                    .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1

            s2s_attn.masked_fill_(attn_mask, 0.0)

            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            t_en = model.text_encoder(texts, input_lengths, text_mask)

            if bool(random.getrandbits(1)):
                asr = t_en @ s2s_attn
            else:
                asr = t_en @ s2s_attn_mono

            mel_input_length_all = accelerator.gather(mel_input_length)
            mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)

            # Skip batches that are too short for randint / style encoder
            if mel_len < 1 or mel_len_st < 1:
                continue
            if int(mel_input_length.min().item() / 2) <= mel_len:
                continue

            en = []
            gt = []
            wav = []
            st = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])

                y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                random_start = np.random.randint(0, max(1, mel_length - mel_len_st))
                st.append(mels[bib, :, (random_start * 2):((random_start + mel_len_st) * 2)])

            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            wav = torch.stack(wav).float().detach()

            # Style encoder needs enough frames after downsampling
            if gt.shape[-1] < 80:
                continue

            with torch.no_grad():
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))

            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            y_rec = model.decoder(en, F0_real, real_norm, s)

            if epoch >= TMA_epoch:
                optimizer.zero_grad()
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                accelerator.backward(d_loss)
                optimizer.step('msd')
                optimizer.step('mpd')
            else:
                d_loss = 0

            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

            if epoch >= TMA_epoch:
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                    loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s /= texts.size(0)

                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                loss_slm = wl(wav.detach(), y_rec).mean()

                g_loss = (
                    loss_params.lambda_mel * loss_mel
                    + loss_params.lambda_mono * loss_mono
                    + loss_params.lambda_s2s * loss_s2s
                    + loss_params.lambda_gen * loss_gen_all
                    + loss_params.lambda_slm * loss_slm
                )
            else:
                loss_s2s = 0
                loss_mono = 0
                loss_gen_all = 0
                loss_slm = 0
                g_loss = loss_mel

            running_loss += accelerator.gather(loss_mel).mean().item()

            def _scalar(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().mean().item())
                return float(x) if x else 0.0

            epoch_mel += accelerator.gather(loss_mel).mean().item()
            epoch_gen += _scalar(loss_gen_all)
            epoch_d += _scalar(d_loss)
            epoch_mono += _scalar(loss_mono)
            epoch_s2s += _scalar(loss_s2s)
            epoch_slm += _scalar(loss_slm)
            epoch_steps += 1

            accelerator.backward(g_loss)

            optimizer.step('text_encoder')
            optimizer.step('style_encoder')
            optimizer.step('decoder')

            if epoch >= TMA_epoch:
                optimizer.step('text_aligner')
                optimizer.step('pitch_extractor')

            iters = iters + 1

            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                log_print(
                    'Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f'
                    % (
                        epoch + 1,
                        epochs,
                        i + 1,
                        max(1, len(train_list) // batch_size),
                        running_loss / log_interval,
                        loss_gen_all,
                        d_loss,
                        loss_mono,
                        loss_s2s,
                        loss_slm,
                    ),
                    logger,
                )

                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/mono_loss', loss_mono, iters)
                writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                writer.add_scalar('train/slm_loss', loss_slm, iters)

                running_loss = 0
                print('Time elasped:', time.time() - start_time, flush=True)

        loss_test = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)

                text_mask = length_to_mask(input_lengths).to(texts.device)
                attn_mask = (
                    (~mask)
                    .unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = attn_mask.float() * (
                    (~text_mask)
                    .unsqueeze(-1)
                    .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1
                s2s_attn.masked_fill_(attn_mask, 0.0)

                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = t_en @ s2s_attn

                mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])
                if mel_len < 1 or int(mel_input_length.min().item() / 2) <= mel_len:
                    continue

                en = []
                gt = []
                wav = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)
                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start:random_start + mel_len])
                    gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                    y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                    wav.append(torch.from_numpy(y).to(device))

                wav = torch.stack(wav).float().detach()
                en = torch.stack(en)
                gt = torch.stack(gt).detach()

                if gt.shape[-1] < 80:
                    continue

                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                s = model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec = model.decoder(en, F0_real, real_norm, s)

                loss_mel = stft_loss(y_rec.squeeze(), wav.detach())
                loss_test += accelerator.gather(loss_mel).mean().item()
                iters_test += 1

        early_stop = False
        if accelerator.is_main_process:
            print('Epochs:', epoch + 1, flush=True)
            eval_loss = (loss_test / iters_test) if iters_test > 0 else float('inf')
            log_print('Validation loss: %.3f' % eval_loss + '\n\n\n\n', logger)
            writer.add_scalar('eval/mel_loss', eval_loss, epoch + 1)

            denom = max(1, epoch_steps)
            train_mel = epoch_mel / denom
            train_gen = epoch_gen / denom
            train_d = epoch_d / denom
            train_mono = epoch_mono / denom
            train_s2s = epoch_s2s / denom
            train_slm = epoch_slm / denom

            if iters_test > 0:
                attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
                writer.add_figure('eval/attn', attn_image, epoch)

                with torch.no_grad():
                    for bib in range(len(asr)):
                        mel_length = int(mel_input_length[bib].item())
                        if mel_length < 80:
                            continue

                        gt = mels[bib, :, :mel_length].unsqueeze(0)
                        en = asr[bib, :, :mel_length // 2].unsqueeze(0)

                        F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                        F0_real = F0_real.unsqueeze(0)
                        s = model.style_encoder(gt.unsqueeze(1))
                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                        y_rec = model.decoder(en, F0_real, real_norm, s)

                        writer.add_audio('eval/y' + str(bib), y_rec.cpu().numpy().squeeze(), epoch, sample_rate=sr)
                        if epoch == 0:
                            writer.add_audio('gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)

                        if bib >= 6:
                            break

            # Checkpoint / early-stop only on save_freq cadence (epochs 10, 20, ...)
            if (epoch + 1) % saving_epoch == 0:
                if is_improved(eval_loss, best_loss, early_stopping_min_delta):
                    best_loss = eval_loss
                    best_epoch = epoch + 1
                    patience = 0
                    print('New best checkpoint — saving...', flush=True)
                    state = {
                        'net': {key: accelerator.unwrap_model(model[key]).state_dict() for key in model},
                        'optimizer': optimizer.state_dict(),
                        'iters': iters,
                        'val_loss': eval_loss,
                        'epoch': epoch,
                    }
                    save_best_checkpoint(
                        log_dir,
                        state,
                        epoch_prefix='1st',
                        epoch=epoch + 1,
                        best_name=first_stage_name,
                    )
                else:
                    patience += 1
                    log_print(
                        f'No val improvement at epoch {epoch + 1} '
                        f'(best={best_loss:.3f} @ epoch {best_epoch}). '
                        f'patience={patience}/{early_stopping_patience}',
                        logger,
                    )
                    if patience >= early_stopping_patience:
                        early_stop = True
                        log_print(
                            f'Early stopping: no val improvement for '
                            f'{early_stopping_patience} intervals.',
                            logger,
                        )

                log_print(
                    f'Eval @ epoch {epoch + 1} | val={eval_loss:.3f} | '
                    f'best={best_loss:.3f} @ epoch {best_epoch} | '
                    f'patience={patience}/{early_stopping_patience}',
                    logger,
                )

            append_epoch_metrics(metrics_csv, {
                'epoch': epoch + 1,
                'train_mel': f'{train_mel:.6f}',
                'val_mel': f'{eval_loss:.6f}' if eval_loss != float('inf') else '',
                'train_gen': f'{train_gen:.6f}',
                'train_d': f'{train_d:.6f}',
                'train_mono': f'{train_mono:.6f}',
                'train_s2s': f'{train_s2s:.6f}',
                'train_slm': f'{train_slm:.6f}',
                'best_val': f'{best_loss:.6f}' if best_loss != float('inf') else '',
                'best_epoch': best_epoch if best_epoch >= 0 else '',
                'patience': patience,
            })

        # Sync early-stop across ranks (no-op for single process)
        stop_flag = torch.tensor(
            [1 if (accelerator.is_main_process and early_stop) else 0],
            device=device,
        )
        if accelerator.num_processes > 1:
            stop_flag = accelerator.reduce(stop_flag, reduction='max')
        if stop_flag.item() > 0:
            break

    if accelerator.is_main_process:
        best_path = osp.join(log_dir, first_stage_name)
        if osp.isfile(best_path):
            log_print(
                f'Stage 1 done. Best val={best_loss:.3f} @ epoch {best_epoch}. '
                f'Checkpoint: {best_path}',
                logger,
            )
        else:
            log_print(
                'Stage 1 done but no best checkpoint was saved '
                '(need at least one save_freq eval with finite val loss).',
                logger,
            )


if __name__ == "__main__":
    main()
