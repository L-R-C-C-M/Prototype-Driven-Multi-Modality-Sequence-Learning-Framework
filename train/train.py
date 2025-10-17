from datetime import datetime
import os
import sys
import logging
import torch
import pdb
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix, balanced_accuracy_score,precision_score
from monai.utils import set_determinism
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_3d.dataset import Whole_NSCLC_PETCT_Dataset
from model.MultiModal_NewDual import PETCTSequenceTransformer
from model.ComparisonFramework import ComparisonFramework
from model.Proto import DynamicPrototype
from model.Loss import FinalLoss

import matplotlib.pyplot as plt

def predict(output, proto):
    output_norm = F.normalize(output, dim=-1)
    proto0_norm = F.normalize(proto["0"], dim=0).reshape(-1)
    proto1_norm = F.normalize(proto["1"], dim=0).reshape(-1)
    predict_list = []
    pred_logits=[]

    for i in range(output_norm.shape[0]):
        output_norm_i = output_norm[i].reshape(-1)
        sim_proto0 = torch.matmul(output_norm_i, proto0_norm)
        sim_proto1 = torch.matmul(output_norm_i, proto1_norm)
        predict = 1 if sim_proto0 < sim_proto1 else 0
        pred_logits.append([sim_proto0.item(),sim_proto1.item()])
        predict_list.append(predict)
    return predict_list,pred_logits


def getscores(labels,predicts,pred_logits):
    labels=np.array(labels)
    predicts=np.array(predicts)
    pred_logits=np.array(pred_logits)
    cm = confusion_matrix(labels, predicts)
    acc=accuracy_score(labels, predicts)
    balanced_acc = balanced_accuracy_score(labels, predicts)
    binary_f1 = f1_score(labels, predicts, average='binary')
    binary_precision = precision_score(labels, predicts, average='binary')
    binary_recall = recall_score(labels, predicts, average='binary')
    macro_f1 = f1_score(labels, predicts, average='macro')
    macro_precision = precision_score(labels, predicts, average='macro')
    macro_recall = recall_score(labels, predicts, average='macro')
    micro_f1 = f1_score(labels, predicts, average='micro')
    micro_precision = precision_score(labels, predicts, average='micro')
    micro_recall = recall_score(labels, predicts, average='micro')
    try:
        pred_probs = torch.softmax(torch.tensor(pred_logits), dim=1).numpy()
        pred_pos_probs = pred_probs[:, 1]  
        auc = roc_auc_score(labels, pred_pos_probs)
    except ValueError:
        auc = float('nan')
    if cm.shape == (2, 2):
        TN, FP = cm[0][0], cm[0][1]
        specificity = TN / (TN + FP + 1e-6) 
    else:
        specificity = float('nan')

    return {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "auc": auc,
        "binary_f1": binary_f1,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "specificity": specificity,
        "confusion_matrix": cm.tolist()
    }

def processimagearray(image_array1):
    image_array1 = image_array1.unsqueeze(2)
    image_array1=image_array1.float()
    return image_array1

def main(args, writer, current_date):
    set_determinism(seed=args.seed)
    '''=================================Record the training configuration===================================================='''
    logging.basicConfig(filename=args.logfile_name,level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(f"Explaination: {args.explaination}")
    logging.info(f"lr_scheduler = CosineAnnealingLR(optimizer, T_max=100)| lr:{args.lr} | batch_size: {args.batch_size}")
    logging.info("Loss: supcon_loss + self.lambda_proto * prototype_loss+focal_loss")
    logging.info(f"arch:{args.arch} | d_model:{args.d_model}")
    logging.info(f"For ComparisonFramework: queue_size:{args.queue_size} | momentum:{args.momentum} | temperature:{args.temperature}")
    logging.info(f"seed:{args.seed} | Kfold seed:{args.kfold_seed}")
    logging.info(f"freeze_proto_epoch: {args.freeze_proto_epoch}") 
    logging.info("====================================================================================")

    full_dataset = Whole_NSCLC_PETCT_Dataset(
        data_root=args.data_root,
        label_csv_path=args.label_csv_path,
        gene=args.which_gene
    )
    all_indices = np.arange(len(full_dataset))
    all_labels = [full_dataset[i][2] for i in range(len(full_dataset))]
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.kfold_seed)
    fold_results = []

    '''=================================Training===================================================='''
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        print(f"\nTraining Fold {fold+1}/{n_splits}")
        logging.info(f"Starting Fold {fold+1}/{n_splits}")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        encoder_q = PETCTSequenceTransformer(arch=args.arch, d_model=args.d_model)
        encoder_k = PETCTSequenceTransformer(arch=args.arch, d_model=args.d_model)
        model = ComparisonFramework(encoder_q=encoder_q, encoder_k=encoder_k, K=args.queue_size, m=args.momentum, T=args.temperature)
        model.to(device)
        criterion = FinalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=100)
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8]) 
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=nw
        )
        
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=nw
        )

        best_val_balanced_acc=0
        bests={}
        patient_list=[]

        for epoch in range(args.epoches):
            cos_sim = torch.nn.CosineSimilarity(dim=0)
            sim = cos_sim(model.prototype_module.getprototypes()["0"], model.prototype_module.getprototypes()["1"])
            logging.info(f"Proto0-1 cosine similarity: {sim.item():.4f}")
            model.train()
            train_bar = tqdm(train_loader, file=sys.stdout)
            total_loss = 0.0
            total_f_loss=0.0
            total_s_loss=0.0
            total_p_loss=0.0
            train_preds = []
            train_labels = []
            train_pred_logitses=[]
            correct=0
            total_samples=0
            
            for step, data in enumerate(train_bar):
                ct_image_array, pet_image_array, gene_label, patient = data
                ct_image_array = ct_image_array.to(device)  
                pet_image_array = pet_image_array.to(device)
                gene_label = gene_label.to(device)

                ct_image_array = processimagearray(ct_image_array)  
                pet_image_array = processimagearray(pet_image_array)

                mb_features, mb_labels,batch_q,logits,labels =model(ct_image_array, pet_image_array, gene_label)

                loss,s_loss,p_loss,f_loss = criterion(mb_features, mb_labels,model.prototype_module.getprototypes(),batch_q,logits,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_s_loss+=s_loss.item()
                total_p_loss+=p_loss.item()
                total_f_loss+=f_loss.item()
                
                with torch.no_grad():
                    train_predictions,train_pred_logits = predict(batch_q, model.prototype_module.getprototypes())
                    train_preds.extend(train_predictions)
                    train_labels.extend(gene_label.cpu().numpy().tolist())
                    train_pred_logitses.extend(train_pred_logits)
                    correct += (train_predictions == gene_label.cpu().numpy()).sum()
                    total_samples += gene_label.size(0)
                
                train_bar.set_description(
                    "train epoch[{}/{}] loss:{:.3f} acc:{:.3f} supcon_loss:{:.3f} prototype_loss:{:.3f} focal_loss:{:.3f}".format(epoch + 1, args.epoches, total_loss / (step + 1), correct / total_samples,total_s_loss / (step + 1),total_p_loss / (step + 1),total_f_loss / (step + 1)))
                if epoch+1 >= args.freeze_proto_epoch:
                    if sim <= 0.5:
                        model.prototype_module.update(batch_q, gene_label)

            
            avg_loss = total_loss / len(train_bar)
            train_scores=getscores(train_labels, train_preds,train_pred_logitses)
            writer.add_scalar(f"Fold{fold+1}/train/Loss", total_loss / len(train_bar), epoch)
            writer.add_scalar(f"Fold{fold+1}/train/Accuracy", train_scores["acc"], epoch)
            writer.add_scalar(f"Fold{fold+1}/train/AUC", train_scores["auc"], epoch)
            logging.info(f"Fold{fold+1} Train Epoch {epoch+1},Scores: %s", json.dumps(train_scores, separators=(',', ':'), ensure_ascii=False))
            logging.info(f"Fold{fold+1} Train Epoch {epoch+1},Loss:{avg_loss}, supcon_loss: {total_s_loss / len(train_bar)}, prototype_loss: {total_p_loss / len(train_bar)}, focal_loss: {total_f_loss / len(train_bar)}")

            
            model.eval()
            val_total_loss = 0.0
            val_preds = []
            val_labels = []
            val_pred_logitses=[]
            val_correct=0
            val_total_samples=0
            val_bar = tqdm(val_loader, desc=f"val epoch[{epoch + 1}/{args.epoches}]", leave=True)
            
            with torch.no_grad():
                for val_step, val_data in enumerate(val_bar):
                    ct_image_array, pet_image_array, val_gene_label, patient = val_data
                    if epoch<1:
                        patient_list.append(patient)
                    
                    ct_image_array = ct_image_array.to(device)
                    pet_image_array = pet_image_array.to(device)
                    val_gene_label = val_gene_label.to(device)
                    
                    ct_image_array = processimagearray(ct_image_array)
                    pet_image_array = processimagearray(pet_image_array)
                    
                    q,val_logits = model.encoder_q(ct_image_array, pet_image_array)

                    val_predict,val_pred_logits=predict(q,model.prototype_module.getprototypes())
                    val_labels.extend(val_gene_label.cpu().numpy().tolist())
                    val_preds.extend(val_predict)
                    val_pred_logitses.extend(val_pred_logits)
                    val_correct += (val_predict == val_gene_label.cpu().numpy()).sum()
                    val_total_samples += val_gene_label.size(0)
                    val_bar.set_description(
                        "val epoch[{}/{}] acc:{:.3f}".format(epoch + 1, args.epoches, val_correct / val_total_samples))

            if epoch<1:
                logging.info(f"{patient_list}")
            val_scores=getscores(val_labels, val_preds,val_pred_logitses)
            writer.add_scalar(f"Fold{fold+1}/val/Accuracy", val_scores["acc"], epoch)
            writer.add_scalar(f"Fold{fold+1}/val/BAcc", val_scores["balanced_acc"], epoch)
            writer.add_scalar(f"Fold{fold+1}/val/AUC", val_scores["auc"], epoch)
            logging.info(f"Fold{fold+1} Val Epoch {epoch+1}, Scores: %s", json.dumps(val_scores, separators=(',', ':'), ensure_ascii=False))

            if best_val_balanced_acc<val_scores["balanced_acc"]:
                best_val_balanced_acc=val_scores["balanced_acc"]
                best_val_auc=val_scores["auc"]
                bests=val_scores.copy()

                torch.save(model.state_dict(), f'{args.model_save_path}_fold{fold+1}.pth')
            elif best_val_balanced_acc==val_scores["balanced_acc"]:
                if best_val_auc<val_scores["auc"]:
                    best_val_auc=val_scores["auc"]
                    bests=val_scores.copy()
                    torch.save(model.state_dict(), f'{args.model_save_path}_fold{fold+1}.pth')
            lr_scheduler.step()
        fold_results.append(bests)

    logging.info(fold_results)
    mean_balanced_acc = sum(fold["balanced_acc"] for fold in fold_results) / len(fold_results)
    logging.info(f"Mean balanced accuracy across all folds: {mean_balanced_acc:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default="NSCLC/3D-PETCT-Processed")
    parser.add_argument('--label_csv_path', type=str, default="NSCLC/labels_cleaned.csv")
    which_gene='egfr'
    parser.add_argument('--which_gene', type=str, default=which_gene)

    parser.add_argument('--epoches', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kfold_seed', type=int, default=42)
    parser.add_argument('--freeze_proto_epoch', type=int, default=0)
    
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--queue_size', type=int, default=2048)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    current_date = datetime.now().strftime('%Y%m%d%H%M%S')
    dir_name=f'./work_dirs/{current_date}_test_{which_gene}'
    os.makedirs(dir_name,exist_ok=True)
    parser.add_argument('--dir_name', type=str, default=dir_name)
    parser.add_argument('--logfile_name', type=str, default=f'{dir_name}/log.log')
    parser.add_argument('--model_save_path', type=str, default=f'{dir_name}/model.pth')
    parser.add_argument('--explaination', type=str, default='')
    writer_dir = os.path.join("runs", dir_name)
    writer = SummaryWriter(writer_dir)
    
    opt = parser.parse_args()
    main(opt, writer, current_date)
    writer.close()