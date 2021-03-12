import os
import time
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import Model
from utils import util, data_utils, evaluate


def train(args):
    MAIN_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL = args.dataset_name + '_NeuMF'
    MODEL_PATH = os.path.join(MAIN_PATH, 'saved_models')
    DATA_PATH = os.path.join(MAIN_PATH, 'data', args.dataset_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    print(f'seeding for reproducibility at {args.seed}...')
    util.seed_everything(args.seed)

    print('loading processed data...')
    dataset = pd.read_csv(
        os.path.join(DATA_PATH, args.dataset_name + '.csv'),
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )

    num_users = dataset['user_id'].nunique() + 1
    num_items = dataset['item_id'].nunique() + 1

    print('constructing train and test datasets and saving splits...')
    data = data_utils.NCFData(args, dataset)
    train_loader = data.get_train_instance()
    test_loader = data.get_test_instance()

    data.train_ratings.to_csv(os.path.join(DATA_PATH, 'train.csv'), header=False, index=False)
    data.test_ratings.to_csv(os.path.join(DATA_PATH, 'test.csv'), header=False, index=False)
    # data.negatives.to_csv(os.path.join(DATA_PATH, 'negatives.csv'), header=False, index=False)

    print('building model and optimizer...')
    model = Model.NeuMF(args, num_users, num_items).to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('training initiated!')
    best_hr = 0
    for epoch in range(1, args.epochs + 1):
        model.train()  # Enable dropout (if have).
        start_time = time.time()

        for user, item, label in tqdm(train_loader):
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/Train_loss', loss.item(), epoch)

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, device)
        writer.add_scalar('Perfomance/HR@10', HR, epoch)
        writer.add_scalar('Perfomance/NDCG@10', NDCG, epoch)

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch} took: {time.strftime('%H: %M: %S', time.gmtime(elapsed_time))} ")
        print("HR@{}: {:.3f}\tNDCG@{}: {:.3f}".format(args.top_k, np.mean(HR), args.top_k, np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(MODEL_PATH):
                    os.mkdir(MODEL_PATH)
                torch.save(model, '{}/{}.pth'.format(MODEL_PATH, MODEL))
                print('model saved!')

    writer.close()
    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MusicalInstruments",
        choices=("InstantVideo", "MusicalInstruments", "MovieLens"),
        help="Name of the Dataset."
    )
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=30,
                        help="training epoches")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--factor_num",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--layers",
                        nargs='+',
                        default=[64, 32, 16, 8],
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument("--num_ng",
                        type=int,
                        default=4,
                        help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test",
                        type=int,
                        default=100,
                        help="Number of negative samples for test set")
    parser.add_argument("--out",
                        default=True,
                        help="save model or not")

    print('parsing args...')
    train(parser.parse_args())
