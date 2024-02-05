import torch
import time
import logging
class AugTools(object):
    def __init__(self, args):
        self.args = args
        self.clip_logits = None
        self.aug_logits = None
        self.tpt_logits = None
        self.logits = None
        self.trained_logits = None
        self.trained_aug_logits = None
        self.clip_results = []
        self.clip_results_acc5 = []
        self.aug_results = []
        self.aug_results_acc5 = []
        self.trained_results = []
        self.trained_results_acc5 = []
        self.trained_aug_results = []
        self.trained_aug_results_acc5 = []
        self.results = []
        self.results_acc5 = []
        for i in range(self.args.tta_steps):
            self.aug_results.append([])
            self.aug_results_acc5.append([])
            self.trained_aug_results.append([])
            self.trained_aug_results_acc5.append([])
        self.target = None
        self.clip_acc = 0
        self.aug_acc = 0
        self.trained_acc = 0
        self.trained_aug_acc = 0
        self.weight = args.aug_weight
        self.init_logger()

    def init_logger(self):
        self.log_path = "./logs-AAAI/"+self.args.info
        self.log_path = self.log_path + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())   
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level = logging.INFO)
        self.handler = logging.FileHandler(self.log_path + '.log')
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        
    def cal_acc(self, output=None, target=None):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            if target is None:
                target = self.target
            if self.args.test_sets in ["A"]:
                logits = (self.aug_logits*self.weight + self.trained_logits*(1-self.weight)) 
            elif self.args.test_sets in ["R"]:
                logits = (self.aug_logits*self.weight + self.trained_logits*(1-self.weight)) 
            elif self.args.test_sets in ["K"]:
                logits = self.trained_aug_logits 
            _, pred = logits.topk(1, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            self.results.append(correct.item())
            _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            if True in correct.reshape(-1):
                result_acc5 = True
            else:
                result_acc5 = False
            self.results_acc5.append(result_acc5)

    def cal_clip(self, output, target=None, topk=1):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            if target is None:
                target = self.target
            output.unsqueeze_(0)
            _, pred = output.topk(1, dim=1, largest=True, sorted=True)
            # print(pred.shape)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            self.clip_results.append(correct.item())
            
            _, pred = output.topk(5, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            if True in correct.reshape(-1):
                result_acc5 = True
            else:
                result_acc5 = False
            self.clip_results_acc5.append(result_acc5)
            return 

    def cal_aug(self, output, j, target=None, topk=1):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            if target is None:
                target = self.target
            if self.args.test_sets in ["A", "R", "K"]:
                output = output.mean(dim=0)
                output.unsqueeze_(0)
                self.aug_logits = output
                _, pred = output.topk(1, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                self.aug_results[j].append(correct.item())
                _, pred = output.topk(5, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                if True in correct.reshape(-1):
                    result_acc5 = True
                else:
                    result_acc5 = False
                self.aug_results_acc5[j].append(result_acc5)
            return
        
    def cal_trained_aug(self, output, j, target=None, topk=1, top10=None):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            if target is None:
                target = self.target
            if self.args.test_sets in ["A", "R", "K"]:
                output = output.mean(dim=0)
                output.unsqueeze_(0)
                self.trained_aug_logits = output
                _, pred = output.topk(1, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                self.trained_aug_results[j].append(correct.item())
                _, pred = output.topk(5, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                if True in correct.reshape(-1):
                    result_acc5 = True
                else:
                    result_acc5 = False
                self.trained_aug_results_acc5[j].append(result_acc5)
            return

    def cal_trained(self, output, target=None, topk=1, top10=None):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            if target is None:
                target = self.target
            if self.args.test_sets in ["A", "R", "K"]:
                output = output.mean(dim=0)
                output.unsqueeze_(0)
                self.trained_logits = output
                _, pred = output.topk(1, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                self.trained_results.append(correct.item())
                _, pred = output.topk(5, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                if True in correct.reshape(-1):
                    result_acc5 = True
                else:
                    result_acc5 = False
                self.trained_results_acc5.append(result_acc5)
            return 
    
    def logger_nums(self, end=False):
        if end:
            self.logger.info(' ')
            self.logger.info(' ')
        all = len(self.results)
        self.logger.info('all test samples number: {}'.format(all))
        correct = sum(self.results)
        correct_rate = correct / all
        self.logger.info('ACC@1: {:.4f}'.format(correct_rate))
        correct_acc5 = sum(self.results_acc5)
        correct_acc5_rate = correct_acc5 / all
        self.logger.info('ACC@5: {:.4f}'.format(correct_acc5_rate))
        print('test samples: {}\tACC@1: {:.4f}\tACC@5: {:.4f}'.format(all, correct_rate, correct_acc5_rate))