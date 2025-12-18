# train.py
import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scipy.stats import spearmanr

from model import CNN3D_LBA
from data import CNN3D_TransformLBA
import resnet

# 用我们在 datasets.py 里新增的构建器：一次性打开 train/val/test LMDB，再用三份 CSV 的 pdb 列过滤
# 注意：请确保你的项目里已经按我给的方式修改了 datasets.py：
#   - CombinedLMDBStore, CombinedLMDBDataset, build_datasets_from_three_csvs
from lba.datasets import build_datasets_from_three_csvs


# 构建 3D CNN 头
def conv_model(in_channels, spatial_size, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2 ** n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1] * int((num_conv + 1) / 2)
    max_pool_sizes = [2] * num_conv
    max_pool_strides = [2] * num_conv
    fc_units = [32]

    model = CNN3D_LBA(
        in_channels, spatial_size,
        args.conv_drop_rate,
        args.fc_drop_rate,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=args.batch_norm,
        dropout=not args.no_dropout)
    return model


def train_loop(pre_model, model, loader, optimizer, device):
    model.train()
    losses = []
    epoch_loss = 0.0
    progress_format = 'train loss: {:6.6f}'
    with tqdm.tqdm(total=len(loader), desc=progress_format.format(0)) as t:
        for i, data in enumerate(loader):
            feature = data['feature'].to(device).to(torch.float32)
            new_feature = pre_model(feature)
            label = data['label'].to(device).to(torch.float32)

            optimizer.zero_grad()
            output = model(new_feature)
            batch_losses = F.mse_loss(output, label, reduction='none')
            batch_losses_mean = batch_losses.mean()
            batch_losses_mean.backward()
            optimizer.step()

            epoch_loss += (batch_losses_mean.item() - epoch_loss) / float(i + 1)
            losses.extend(batch_losses.tolist())
            t.set_description(progress_format.format(np.sqrt(epoch_loss)))
            t.update(1)

    return np.sqrt(np.mean(losses))


@torch.no_grad()
def test(pre_model, model, loader, device):
    model.eval()
    losses = []
    abs_errors = []
    ids, y_true, y_pred = [], [], []

    for data in loader:
        feature = data['feature'].to(device).to(torch.float32)
        new_feature = pre_model(feature)
        label = data['label'].to(device).to(torch.float32)
        output = model(new_feature)

        batch_losses = F.mse_loss(output, label, reduction='none')
        losses.extend(batch_losses.tolist())
        
        batch_abs_errors = F.l1_loss(output, label, reduction='none')
        abs_errors.extend(batch_abs_errors.tolist())
        
        ids.extend(data['id'])
        y_true.extend(label.tolist())
        y_pred.extend(output.tolist())

    results_df = pd.DataFrame(
        np.array([ids, y_true, y_pred]).T,
        columns=['structure', 'true', 'pred'],
    )
    r_p = np.corrcoef(y_true, y_pred)[0, 1]
    r_s = spearmanr(y_true, y_pred)[0]
    rmse = np.sqrt(np.mean(losses))
    mae = np.mean(abs_errors)
    return rmse, mae, r_p, r_s, results_df


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def build_transform(split_name, seed):
    # 和你原来的用法一致：每个 split 都传一个 CNN3D_TransformLBA(random_seed=...)
    return CNN3D_TransformLBA(random_seed=seed)


def train(args, device, test_mode=False):
    print("Training model with config:")
    print(str(json.dumps(args.__dict__, indent=4)) + "\n")

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # === 核心更改：一次性打开 train/val/test 三个 LMDB，并用三份 CSV 的 pdb 列过滤 ===
    train_dataset, val_dataset, test_dataset, store = build_datasets_from_three_csvs(
        data_dir=args.data_dir,
        csv_train=args.csv_train,
        csv_val=args.csv_val,
        csv_test=args.csv_test,
        pdb_col=args.pdb_col,
        transform_factory=build_transform,
        random_seed=args.random_seed,
    )

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # 看一下特征维度（和你原来的打印保持一致）
    for data in train_loader:
        print(data['feature'].size())
        break

    # 预训练 backbone
    pre_model = resnet.generate_model(18).to(device)
    pre_model.load_state_dict(torch.load('model_stage1_epoch20.pth'), strict=False)  # 自行准备权重文件
    in_channels = 32
    spatial_size = 23

    # 3D CNN 头
    model = conv_model(in_channels, spatial_size, args).to(device)
    print(model)

    best_val_loss = np.inf
    best_rp = 0.0
    best_rs = 0.0
    best_mae = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        # Debug: Print first batch info
        if epoch == 1:
            for i, data in enumerate(train_loader):
                print(f"DEBUG: Batch {i} feature shape: {data['feature'].shape}")
                print(f"DEBUG: Batch {i} labels: {data['label']}")
                feature = data['feature'].to(device).to(torch.float32)
                new_feature = pre_model(feature)
                output = model(new_feature)
                print(f"DEBUG: Batch {i} output: {output.flatten()[:5]}")
                loss = F.mse_loss(output, data['label'].to(device).to(torch.float32), reduction='none')
                print(f"DEBUG: Batch {i} raw loss: {loss.mean().item()}")
                break

        train_loss = train_loop(pre_model, model, train_loader, optimizer, device)
        val_loss, val_mae, r_p, r_s, val_df = test(pre_model, model, val_loader, device)

        if val_loss < best_val_loss:
            print(f"\nSave model at epoch {epoch:03d}, val_loss: {val_loss:.4f}")
            save_weights(model, os.path.join(args.output_dir, f'best_weights.pt'))
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
            best_mae = val_mae

        elapsed = (time.time() - start)
        print('Epoch {:03d} finished in : {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Val MAE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
            train_loss, val_loss, val_mae, r_p, r_s))

    if test_mode:
        print(f"Loading best weights from {os.path.join(args.output_dir, 'best_weights.pt')}")
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
        rmse, mae, pearson, spearman, test_df = test(pre_model, model, test_loader, device)
        
        # Save predictions
        test_df.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)
        test_df.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
        
        print('Test RMSE: {:.7f}, Test MAE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
            rmse, mae, pearson, spearman))
        
        test_file = os.path.join(args.output_dir, f'test_metrics.txt')
        with open(test_file, 'a+') as out:
            out.write('Seed\tRMSE\tMAE\tPearson\tSpearman\n')
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(
                args.random_seed, rmse, mae, pearson, spearman))

    # 训练/测试结束后，可关闭 LMDB 资源
    try:
        store.close()
    except Exception:
        pass

    return best_val_loss, best_mae, best_rp, best_rs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='包含 train/val/test 三个 LMDB 目录的根目录')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str, default='output_train')
    parser.add_argument('--unobserved', action='store_true', default=False)

    # 新增：三份 CSV 路径 + 列名
    parser.add_argument('--csv_train', type=str, default=r"C:\Users\Administrator\Desktop\RNA\train.csv", help='训练集 CSV（含列 pdb）')
    parser.add_argument('--csv_val',   type=str, default=r"C:\Users\Administrator\Desktop\RNA\val.csv", help='验证集 CSV（含列 pdb）')
    parser.add_argument('--csv_test',  type=str, default=r"C:\Users\Administrator\Desktop\RNA\test.csv", help='测试集 CSV（含列 pdb）')
    parser.add_argument('--pdb_col',   type=str, default='pdb', help='CSV 里作为样本 ID 的列名')

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=int(1))

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output dir（保留你的原始逻辑）
    args.output_dir = os.path.join(args.output_dir, 'output')
    assert args.output_dir is not None
    if args.unobserved:
        args.output_dir = os.path.join(args.output_dir, 'None')
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        num = 0
        while True:
            dirpath = os.path.join(args.output_dir, str(num))
            if os.path.exists(dirpath):
                num += 1
            else:
                args.output_dir = dirpath
                print('Creating output directory {:}'.format(args.output_dir))
                os.makedirs(args.output_dir)
                break

    print(f"Running mode {args.mode:} with seed {args.random_seed:} and output dir {args.output_dir}")
    train(args, device, args.mode == 'test')
