import numpy as np
import torch
import random,string
import datetime
import os
import pandas as pd
class SelectModel:# inputs: epoch, outputs, model_save_pth

    def __init__(self, save_model_pth=None):
        self.fname = os.path.join(save_model_pth, 'best_classifier.model')
        self.dataframe = pd.DataFrame()
        self.max_epoch = 0




    def __call__(self, epoch, outputs, classifier_model):
        if epoch < 4:
            self.dataframe = self.dataframe.append([[epoch + 1, outputs[2], -outputs[3]]], ignore_index=True)  # epoch, loss_val, acc_val
            if epoch == 3:
                self.dataframe.columns = ['epoch', 'loss_val', '-acc_val']
            classifier_model.load_state_dict(torch.load(self.fname))
            self.max_epoch = epoch + 1
            name = self.fname + '_' + str(epoch + 1)
            torch.save(classifier_model.state_dict(), name)
                # classifier_model.load_state_dict(early_stopping.load_checkpoint())
        else:
            df2 = pd.DataFrame({'epoch': [epoch + 1], 'loss_val': [outputs[2]], '-acc_val': [-outputs[3]]})
            self.dataframe = self.dataframe.append(df2, ignore_index=True)
            self.dataframe['loss_rank'] = 1 / self.dataframe['loss_val'].rank()
            self.dataframe['acc_rank'] = 1 / self.dataframe['-acc_val'].rank()
            self.dataframe['ave_rank'] = self.dataframe[['loss_rank', 'acc_rank']].apply(lambda x: (x['loss_rank'] + x['acc_rank']) / 2,
                                                                               axis=1)
            print(self.dataframe)
            column1 = self.dataframe.loc[:, "ave_rank"]
            column1_argmin = column1[column1 == column1.min()].index
            column1_argmin = np.random.choice(column1_argmin)
            current_rank = self.dataframe.at[self.dataframe[self.dataframe.epoch == epoch + 1].index.tolist()[0], 'ave_rank']
            min_rank = self.dataframe.at[column1_argmin, 'ave_rank']

            if current_rank.astype(int) >= min_rank.astype(int):
                self.dataframe = self.dataframe.drop(index=[column1_argmin])
                name = self.fname + '_' + str(epoch + 1)
                torch.save(classifier_model.state_dict(), name)
                print('----drop the row ranked at bottom', self.dataframe)
            else:
                self.dataframe = self.dataframe.drop(index=[epoch])
                print('----drop current row', self.dataframe)

            column1_argmax = column1[column1 == column1.max()].index.tolist()
            self.max_epoch = self.dataframe.at[column1_argmax[-1], 'epoch']
            # print(column1_argmax, column1_argmax[-1], self.max_epoch)
            load_name = self.fname + '_' + str(self.max_epoch)
            classifier_model.load_state_dict(torch.load(load_name))

        return self.max_epoch, classifier_model