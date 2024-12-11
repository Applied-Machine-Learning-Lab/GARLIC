# -*- coding: utf-8 -*-
# @Time : 2024/5/8 16:35
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : model_training.py
# @Software: PyCharm


import logging
import random
import time

import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle as pkl
from config import parser
from model.DSEmb.graphbase import NCModel
from model.RL.DDT import DDT
from model.RL.layers import Reward, Actor, Critic
from utils.clear_model_logs import clear
from utils.data_utils import DS_data, TrajData, TrajAllData
from utils.eval_utils import acc_f1, action_accuracy
from utils.train_utils import format_metrics, select_action

if __name__ == "__main__":
    tmp_path = '../'

    clear(tmp_path)

    logging.getLogger().setLevel(logging.ERROR)

    args = parser.parse_args()

    args.log_dir = tmp_path + args.log_dir
    args.trajectory_pos = tmp_path + args.trajectory_pos
    args.trajectory_neg = tmp_path + args.trajectory_neg
    args.trajectory_all = tmp_path + args.trajectory_all
    args.multigraph_adj = tmp_path + args.multigraph_adj
    args.multigraph_fea = tmp_path + args.multigraph_fea

    args.device = torch.device("cpu")
    if args.cuda >= 0:
        args.device = torch.device("cuda:" + str(args.cuda))

    ID = 0
    # tensorboardX logbook
    Writer = SummaryWriter(args.log_dir + '/tensorboard/' +  str(ID) + '/')

    # read data
    ## graph data ##
    DSDATA = DS_data(args)
    DS_dataloader = DataLoader(DSDATA, batch_size=4, shuffle=True, num_workers=1)
    ### get node number, node feature dimension
    for i, batch in enumerate(DS_dataloader):
        a, b, c, a_label, b_label, c_label = batch
        args.n_nodes = a.shape[1]
        args.feat_dim = a.shape[2]
        break
    adj_G2, adj_G5, adj_G10 = DSDATA.get_adj()
    fea_G2, fea_G5, fea_G10 = DSDATA.get_all_fea()
    ## trajectory data
    contrast_mode_str = 'all'
    trajdata = TrajAllData(args)
    Traj_dataloader = DataLoader(trajdata, batch_size=args.batch_size, shuffle=True, num_workers=1)
    ### get feature dimension, contrast output dimension
    args.in_dim = -1
    if contrast_mode_str == 'all':
        args.out_dim = 50
        args.avg = 'micro'
    else:
        args.out_dim = 2
        args.avg = 'binary'
    for i, batch in enumerate(Traj_dataloader):
        vid, state, action, reward = batch
        args.in_dim = state.shape[2] + action.shape[2]
        break
    ## positive Trajectory data
    f = open(args.trajectory_pos, 'rb')
    pos_list = pkl.load(f)
    f.close()
    MAX_FEE = 1.0
    for i in range(len(pos_list)):
        if i == 0:
            data = pos_list[i]['data']
            label = pos_list[i]['label']
            static_reward = pos_list[i]['reward']
            MAX_FEE = max(static_reward)
        else:
            data = np.r_[data, pos_list[i]['data']]
            label = np.r_[label, pos_list[i]['label']]
            static_reward = np.r_[static_reward, pos_list[i]['reward']]
            if MAX_FEE < max(static_reward):
                MAX_FEE = max(static_reward)
    # print('MAX_FEE:', MAX_FEE)
    # print('data shape:', data.shape)
    # print('label shape:', label.shape)
    # print('static_reward shape:', static_reward.shape)
    data = torch.tensor(data)
    label = torch.tensor(label)
    static_reward = torch.tensor(static_reward)
    args.num_state = int(data.shape[-1] + args.graph_dim * (adj_G2.shape[0]   + adj_G5.shape[0] + adj_G10.shape[0]))
    args.num_ation = int(label.shape[-1] - 1)
    print('num state:', args.num_state)
    print('num ation:', args.num_ation)

    traj_data_list = []
    label_list = []
    static_reward_list = []
    MAX_FEE_list = []
    # multicar trajectory data
    for carid in range(args.max_car_num):
        ## positive Trajectory data
        f = open(tmp_path+'data/hangzhou/one_traj/' + str(carid) + '.pkl', 'rb')
        data_list = pkl.load(f)
        f.close()
        MAX_FEE = 1.0
        for i in range(len(data_list)):
            if i == 0:
                tmp_data = data_list[i]['data']
                tmp_label = data_list[i]['label']
                tmp_static_reward = data_list[i]['reward']
                tmp_MAX_FEE = max(tmp_static_reward)
            else:
                tmp_data = np.r_[tmp_data, data_list[i]['data']]
                tmp_label = np.r_[tmp_label, data_list[i]['label']]
                tmp_static_reward = np.r_[tmp_static_reward, data_list[i]['reward']]
                if tmp_MAX_FEE < max(tmp_static_reward):
                    tmp_MAX_FEE = max(tmp_static_reward)
        tmp_data = torch.tensor(tmp_data)
        tmp_label = torch.tensor(tmp_label)
        tmp_static_reward = torch.tensor(tmp_static_reward)

        if not args.cuda == -1:
            tmp_data = tmp_data.to(args.device)
            tmp_label = tmp_label.to(args.device)
            tmp_static_reward = tmp_static_reward.to(args.device)
        traj_data_list.append(tmp_data)
        label_list.append(tmp_label)
        static_reward_list.append(tmp_static_reward)
        MAX_FEE_list.append(tmp_MAX_FEE)


    # define graph embedding model
    model_G2 = NCModel(args)
    model_G5 = NCModel(args)
    model_G10 = NCModel(args)
    model_r = Reward(args.in_dim, args.dim, args.out_dim, batch_size=args.batch_size)
    model_ddt = DDT(state_dim=args.num_state,
                    act_dim=args.num_ation,
                    max_length=args.max_len,
                    max_ep_len=150, # maximum length of a trajectory
                    hidden_size=args.dim,
                    n_layer=args.num_layers,
                    n_head=args.n_heads,
                    n_inner=4 * args.dim,
                    activation_function=args.act_gpt,
                    n_positions=256,
                    resid_pdrop=args.dropout,
                    attn_pdrop=args.dropout,
                    )


    # model & data to device
    if not args.cuda == -1:
        model_G2 = model_G2.to(args.device)
        model_G5 = model_G5.to(args.device)
        model_G10 = model_G10.to(args.device)
        model_r = model_r.to(args.device)
        model_ddt = model_ddt.to(args.device)

        adj_G2 = adj_G2.to(args.device)
        adj_G5 = adj_G5.to(args.device)
        adj_G10 = adj_G10.to(args.device)
        fea_G2 = fea_G2.to(args.device)
        fea_G5 = fea_G5.to(args.device)
        fea_G10 = fea_G10.to(args.device)

    # define learning rate reduce frequency
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # define optimizer
    opt_G2 = torch.optim.Adam(params=model_G2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_G5 = torch.optim.Adam(params=model_G5.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_G10 = torch.optim.Adam(params=model_G10.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_r = torch.optim.Adam(params=model_r.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_ddt = torch.optim.Adam(params=model_ddt.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler_G2 = torch.optim.lr_scheduler.StepLR(opt_G2,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_G5 = torch.optim.lr_scheduler.StepLR(opt_G5,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_G10 = torch.optim.lr_scheduler.StepLR(opt_G10,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_r = torch.optim.lr_scheduler.StepLR(opt_r,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_ddt = torch.optim.lr_scheduler.LambdaLR(opt_ddt, lambda steps: min((steps + 1) / args.warm_steps, 1))

    # define loss function
    CEloss = nn.CrossEntropyLoss()
    BCEloss = nn.BCELoss()
    MSEloss = nn.MSELoss()
    # action_loss = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2) # MSEloss(a_hat[:,1], a[:,1]) + BCEloss(a_hat[:,0], torch.round(a[:,0]))
    # action_loss = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(torch.minimum(torch.minimum( (a_hat - a) ** 2, ((a_hat+1) - a) ** 2 ), (a_hat - (a+1)) ** 2))
    action_loss = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
        torch.minimum(torch.minimum((a_hat[:,1] - a[:,1]) ** 2, ((a_hat[:,1] + 1) - a[:,1]) ** 2), (a_hat[:,1] - (a[:,1] + 1)) ** 2) + (a_hat[:,0] - a[:,0]) ** 2)

    global_training_step = 0
    tmp_losses = []
    tmp_accs = []
    for ep in range(args.epochs):
        print('Epoch:', str(ep+1).zfill(4))

        # graph embedding part
        print('Graph embedding...')
        for i, batch in enumerate(DS_dataloader):
            t = time.time()
            a, b, c, a_label, b_label, c_label = batch
            if not args.cuda == -1:
                a = a.to(args.device)
                b = b.to(args.device)
                c = c.to(args.device)
                a_label = a_label.to(args.device)
                b_label = b_label.to(args.device)
                c_label = c_label.to(args.device)

            model_G2.train()
            opt_G2.zero_grad()
            embeddings_G2 = model_G2.encode(a, adj_G2)
            # print('test train emb shape:', embeddings.shape)
            train_metrics_g2 = model_G2.compute_metrics(embeddings_G2, adj_G2, a_label)
            train_metrics_g2['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_G2.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_G2.step()
            lr_scheduler_G2.step()

            model_G5.train()
            opt_G5.zero_grad()
            embeddings_G5 = model_G5.encode(b, adj_G5)
            # print('test train emb shape:', embeddings.shape)
            train_metrics_g5 = model_G5.compute_metrics(embeddings_G5, adj_G5, b_label)
            train_metrics_g5['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_G5.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_G5.step()
            lr_scheduler_G5.step()

            model_G10.train()
            opt_G10.zero_grad()
            embeddings_G10 = model_G10.encode(c, adj_G10)
            # print('test train emb shape:', embeddings.shape)
            train_metrics_g10 = model_G10.compute_metrics(embeddings_G10, adj_G10, c_label)
            train_metrics_g10['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_G10.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_G10.step()
            lr_scheduler_G10.step()

            if (i + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(i + 1),
                                       'lr_G2: {}'.format(lr_scheduler_G2.get_last_lr()[0]),
                                       'lr_G5: {}'.format(lr_scheduler_G5.get_last_lr()[0]),
                                       'lr_G10: {}'.format(lr_scheduler_G10.get_last_lr()[0]),
                                       format_metrics(train_metrics_g2, 'train'),
                                       format_metrics(train_metrics_g5, 'train'),
                                       format_metrics(train_metrics_g10, 'train'),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))
        model_G10.eval()
        model_G5.eval()
        model_G2.eval()
        print('Graph embedding finished!')
        # got embeddings_G2, embeddings_G5 and embeddings_G10
        # print('embedding G2 shape:', embeddings_G2.shape)
        # print('embedding G5 shape:', embeddings_G5.shape)
        # print('embedding G10 shape:', embeddings_G10.shape)

        # dynamic reward part
        print('Dynamic reward function updating...')
        loss = 0
        acc = 0
        f1 = 0
        cnt = 0
        for i, batch in enumerate(Traj_dataloader):
            t = time.time()
            vid, state, action, reward = batch
            # print('vid:', vid)
            if state.shape[0] < args.batch_size:
                continue
            hidden = model_r.init_hidden()
            if not args.cuda == -1:
                vid = vid.to(args.device)
                state = state.to(args.device)
                action = action.to(args.device)
                hidden = hidden.to(args.device)
            x = torch.cat([state, action], dim=2).float()
            vid = vid.long()

            model_r.train()
            opt_r.zero_grad()
            pred, _ = model_r.forward(x, hidden)
            tmp_loss = CEloss(pred, vid)
            tmp_acc, tmp_f1 = acc_f1(pred, vid, average=args.avg)
            loss += tmp_loss
            acc += tmp_acc
            f1 += tmp_f1
            cnt += 1
            loss.backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_r.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_r.step()
            lr_scheduler_r.step()

            if (i + 1) % args.log_freq == 0:
                Writer.add_scalar('Dynamic Reward Accuracy', f1/cnt, global_step=global_training_step)
                Writer.add_scalar('Dynamic Reward loss', loss / cnt, global_step=global_training_step)
                logging.info(" ".join(['Epoch: {:04d}'.format(i + 1),
                                       'lr: {}'.format(lr_scheduler_r.get_last_lr()[0]),
                                       'loss: {}'.format(loss / cnt),
                                       'acc: {}'.format(acc / cnt),
                                       'f1: {}'.format(f1 / cnt),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))
                acc = 0
                f1 = 0
                loss = 0
                cnt = 0
        print('Dynamic reward function updated!')

        for carid in range(args.max_car_num):
            data = traj_data_list[carid]
            # print(data.shape)
            label = label_list[carid]
            static_reward = static_reward_list[carid]
            MAX_FEE = MAX_FEE_list[carid]

            # get dynamcic reward
            model_r.eval()
            print('Batch data calculating...')
            R = 0
            Gt = []
            for ind in range(len(static_reward)-1, -1, -1):
                state = data[ind].unsqueeze(0).unsqueeze(0)
                # print('test state shape:', state.shape)
                true_action = label[ind].unsqueeze(0).unsqueeze(0)
                # print('test action shape:', true_action.shape)
                x = torch.cat([state, true_action], dim=2).float()
                hidden = model_r.init_hidden(1).to(args.device)
                pred, _ = model_r.forward(x, hidden)
                pred = pred.squeeze()
                # print('test pred shape:', pred.shape)
                alpha = 0.05
                R = alpha * (static_reward[ind] / MAX_FEE) + (1 - alpha) * pred[carid] + args.reward_gamma * R
                Gt.insert(0, torch.nan_to_num(R))
            Gt = torch.tensor(Gt, dtype=torch.float).to(args.device)
            # print('R shape:', Gt.shape)

            T_MAX = 144 # Gt.shape[0]
            t_start = random.randint(0, T_MAX - 1)
            print('t_start:', t_start)
            state_dim = -1
            action_vec_dim = -1
            TMOD = T_MAX
            for tt in range(args.update_freq):
                for bs in range(args.batch_size):
                    timestep = []
                    attention_mask = []
                    for ind in range(args.max_len):
                        t = t_start + ind
                        if t % TMOD < T_MAX:
                            tt = time.time()
                            graph_fea_G2 = fea_G2[t].unsqueeze(0)
                            graph_fea_G5 = fea_G5[t].unsqueeze(0)
                            graph_fea_G10 = fea_G10[t].unsqueeze(0)

                            G2_emb = model_G2.encode(graph_fea_G2, adj_G2).detach()
                            G5_emb = model_G5.encode(graph_fea_G5, adj_G5).detach()
                            G10_emb = model_G10.encode(graph_fea_G10, adj_G10).detach()
                            G2_emb = G2_emb.view(1, -1)
                            G5_emb = G5_emb.view(1, -1)
                            G10_emb = G10_emb.view(1, -1)

                            tmp_state = data[t].unsqueeze(0).to(torch.float32)

                            if ind == 0:
                                true_action_vec = label[t][:2].unsqueeze(0)
                                true_action = label[t][2].unsqueeze(0)
                                state = torch.concatenate([tmp_state, G2_emb, G5_emb, G10_emb], dim=1)
                                # state = torch.concatenate([tmp_state, G2_emb], dim=1)
                                R = Gt[t].view(-1, 1)

                                # print('state shape:', state.shape)
                                state_dim = state.shape[1]
                                # print('R shape:', R.shape)
                                # print('action shape:', true_action_vec.shape)
                                action_vec_dim = true_action_vec.shape[1]
                                # print('action shape:', true_action.shape)
                                timestep.append(t % TMOD)
                                attention_mask.append(1.0)
                            else:
                                true_action_vec = torch.concatenate([true_action_vec, label[t][:2].unsqueeze(0)], dim=0)
                                true_action = torch.concatenate([true_action, label[t][2].unsqueeze(0)], dim=0)
                                tmp_state2 = torch.concatenate([tmp_state, G2_emb, G5_emb, G10_emb], dim=1)
                                # tmp_state2 = torch.concatenate([tmp_state, G2_emb], dim=1)
                                state = torch.concatenate([state, tmp_state2], dim=0)
                                R = torch.concatenate([R, Gt[t].view(-1, 1)], dim=0)
                                timestep.append(t % TMOD)
                                attention_mask.append(1.0)

                                # print('state shape (add):', state.shape)
                                # print('R shape (add):', R.shape)
                                # print('action shape (add):', true_action.shape)
                        else:
                            tmp_len = int(args.max_len - ind)
                            true_action_vec = torch.concatenate([true_action_vec, torch.zeros(tmp_len,action_vec_dim).to(args.device)], dim=0)
                            true_action = torch.concatenate([true_action, torch.zeros(tmp_len).to(args.device)], dim=0)
                            state = torch.concatenate([state, torch.zeros(tmp_len, state_dim).to(args.device)], dim=0)
                            R = torch.concatenate([R, torch.zeros(tmp_len,1).to(args.device)], dim=0)
                            for _ in range(tmp_len):
                                timestep.append(0)
                                attention_mask.append(0.0)

                            # print('state shape (else):', state.shape)
                            # print('R shape (else):', R.shape)
                            # print('action shape (else):', true_action_vec.shape)
                            break
                    if bs == 0:
                        bs_state = state.unsqueeze(0)
                        bs_R = R.unsqueeze(0)
                        bs_true_action_vec = true_action_vec.unsqueeze(0)
                        bs_true_action = true_action.unsqueeze(0)
                        bs_timestep = torch.tensor(timestep).unsqueeze(0).to(args.device)
                        bs_attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(args.device)
                    else:
                        bs_state = torch.concatenate([bs_state, state.unsqueeze(0)], dim=0)
                        bs_R = torch.concatenate([bs_R, R.unsqueeze(0)], dim=0)
                        bs_true_action_vec = torch.concatenate([bs_true_action_vec, true_action_vec.unsqueeze(0)], dim=0).float()
                        bs_true_action = torch.concatenate([bs_true_action, true_action.unsqueeze(0)], dim=0)
                        bs_timestep = torch.concatenate([bs_timestep, torch.tensor(timestep).unsqueeze(0).to(args.device)], dim=0)
                        bs_attention_mask = torch.concatenate([bs_attention_mask, torch.tensor(attention_mask).unsqueeze(0).to(args.device)], dim=0)

                # print('batch state shape:', bs_state.shape)
                # print('batch R shape:', bs_R.shape)
                # print('batch action shape:', bs_true_action_vec.shape)
                # print('batch timestep shape:', bs_timestep.shape)
                # print('batch attention_mask shape:', bs_attention_mask.shape)

                t = time.time()

                action_target = torch.clone(bs_true_action_vec)
                action_compare = torch.clone(bs_true_action)

                state_preds, action_preds, reward_preds = model_ddt.forward(
                    bs_state, bs_true_action_vec, bs_R, bs_timestep, attention_mask=bs_attention_mask
                )

                # print('pre_traininig_test action compare:', action_preds[0, 0], action_target[0, 0])

                action_preds = action_preds.reshape(-1, args.num_ation)[bs_attention_mask.reshape(-1) > 0]
                action_target = action_target.reshape(-1, args.num_ation)[bs_attention_mask.reshape(-1) > 0]
                action_compare = action_compare.reshape(-1)[bs_attention_mask.reshape(-1) > 0]

                loss = action_loss(
                    None, action_preds, None,
                    None, action_target, None,
                )

                opt_ddt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_ddt.parameters(), .25)
                opt_ddt.step()

                with torch.no_grad():
                    acc = action_accuracy(action_preds, action_compare)
                    Writer.add_scalar('DDT Accuracy:', acc, global_step=global_training_step)
                    if acc >= 0:
                        tmp_accs.append(acc)
                    loss_value = loss.detach().cpu().item()
                    tmp_losses.append(loss_value)
                    Writer.add_scalar('DDT loss:', loss_value, global_step=global_training_step)
                    if (i + 1) % args.log_freq == 0:
                        logging.info(" ".join(['Epoch: {:04d}'.format(global_training_step + 1),
                                               'DDT Loss: {}'.format(loss_value),
                                               'DDT Accuracy: {}'.format(acc),
                                               'time: {:.4f}s'.format(time.time() - t)
                                               ]))

                global_training_step += 1
            print('DDT loss mean:', np.mean(tmp_losses))
            print('DDT loss std:', np.std(tmp_losses))
            print('DDT acc mean:', np.mean(tmp_accs))
