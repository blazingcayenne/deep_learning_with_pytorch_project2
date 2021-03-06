# # <font style="color:blue">Visualizer Base Class</font>

from abc import ABC, abstractmethod


class Visualizer(ABC):
    @abstractmethod
    def update_charts(self, train_metric, train_loss, test_metric, test_loss, learning_rate, epoch):
        pass

    @abstractmethod
    def add_image(self, tag, image):
        pass
    
    @abstractmethod
    def add_graph(self, model, images):
        pass
    
    @abstractmethod
    def add_figure(self, tag, figure, close=True):
        pass
    
    @abstractmethod
    def add_pr_curves(self, classes, targets, pred_probs):
        pass