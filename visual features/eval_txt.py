import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support


class Eval():
    def __init__(self):
        self.exp_name = 'swap_epoch_v2_valid_set2'
        self.results_folder = os.path.join('results', self.exp_name)
        self.gt_folder = '/mnt/data3/jinyue/dataset/opensource/AffWild2/Annotations/annotations/Annotations/annotations/Annotations/all_annotations'

    def eval_expr(self):
        predicted = []
        true_labels = []
        results_folder = os.path.join(self.results_folder, 'EXPR_Set')
        for txt_file in os.listdir(results_folder):
            with open(os.path.join(results_folder, txt_file), 'r') as f_pred:
                preds = f_pred.readlines()[1:]
                # print(len(preds))

            with open(os.path.join(self.gt_folder, 'EXPR_Set', txt_file), 'r') as f_gt:
                for idx, line in enumerate(f_gt.readlines()[1:]):
                    gt = int(line.strip())
                    if idx < len(preds):
                        if gt != -1 and int(preds[idx].strip()) != -1:
                            true_labels.append(gt)
                            predicted.append(int(preds[idx].strip()))
        acc = accuracy_score(true_labels, predicted)
        total_f1 = f1_score(true_labels, predicted, average='macro')
        print(acc, total_f1)
        return acc, total_f1

    # def eval_au(self, eval_loader):
    #     self.model.eval()
    #     predicted = []
    #     true_labels = []
    #     correct = 0
    #     total = 0
    #     for imgs, labels in tqdm(iter(eval_loader)):
    #         imgs = imgs.to(self.conf.device)
    #         labels = labels.to(self.conf.device)
    #         logits = self.model(imgs)[1]
    #         predicts = torch.greater(logits, 0).type_as(labels)
    #         cmp = predicts.eq(labels).cpu().numpy()
    #         correct += cmp.sum()
    #         total += len(cmp) * 12
    #         predicted += predicts.cpu().numpy().tolist()
    #         true_labels += labels.cpu().numpy().tolist()

    #     acc = correct / total
    #     print('acc:', acc)
    #     total_f1 = f1_score(true_labels, predicted, average='macro')
    #     print('f1:', total_f1)
    #     class_names = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12',
    #                    'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    #     print(classification_report(true_labels,
    #           predicted, target_names=class_names))
    #     return acc, total_f1


if __name__ == '__main__':
    eval_module = Eval()
    eval_module.eval_expr()
    # eval_module.eval_au()
