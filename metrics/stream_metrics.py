import numpy as np
from sklearn.metrics import confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, background_class):
        self.n_classes = n_classes
        self.background_class = background_class
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    # Hyun Edit 밑의 fast_hist_hyun을 활용하기 위한 코드, class label 정보들이 들어감
    def update_hyun(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            lpf = lp.flatten()
            ulp = np.unique(lpf)
            
            # remove background class (sorted as zero)
            ulp_nbg = ulp[ulp!=self.background_class]
            target_class = ulp_nbg[0] # only single class remains
            self.confusion_matrix += self._fast_hist_hyun(lt.flatten(), lpf, target_class)
        
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    # 사실상 confusion matrix를 매 이미지마다 뽑아내는 코드 => 이걸 추후 update에 해서 전체 형태를 summarize하게 됨
    # 여기에 target category를 넣는다?
    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist
    
    # 위의 형태에서 이미지당 특징 카테고리만 탐지하기 위한 형태
    def _fast_hist_hyun(self, label_true, label_pred, target_class):
        # 255를 제거하기 위한 행동
        # mask = (label_true >= 0) & (label_true < self.n_classes)
        # 기존의 mask (255)에 추가로 특정 class에 해당하는 부분만 놔두기
        mask = (label_true >= 0) & (label_true < self.n_classes) & ((label_true==target_class)|(label_pred==target_class))
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist
   
    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
                
        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
    
     # Hyun Edit : Batch 단위로 들어가지 않나? 추가확인이 필요함 - 그렇지 않을경우, 많은 문제가 해결됨 => main_syn 등에서 validation 코드 돌려서 \
        # 추가확인이 필요함 / evaluate using histogram
    def get_results_hyun(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        # x axis add eg [[1,2], [4,5]] => (0.33, 0.555)
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        
        # iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) # zero division
        a = np.diag(hist)
        b = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iu = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        
        mean_iu = np.nanmean(iu)
                
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        
        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
