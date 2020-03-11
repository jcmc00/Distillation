import torch
import torch.nn as nn
from functools import partial

class DistillationLearner():
    def __init__(self, teacher, student, data, loss_fn, opt):
        self.teacher = teacher
        self.student = student
        self.data = data
        self.loss_fn = loss_fn
        self.opt = opt

    def fit(self, epochs):
        self.student.train()
        self.teacher.eval()
        teacher_output = self.get_teacher_output(self.teacher, self.data.train_dl)

        for epoch in range(epochs):
            for i, (xb, yb) in enumerate(self.data.train_dl):
                
                tb = teacher_output[i]
                # expects a partial loss fn
                loss = self.loss_fn(self.student(xb), yb, tb)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            with torch.no_grad():
                tot_loss = 0.
                for i, (xb, yb) in enumerate(self.data.valid_dl):
                    tb = teacher_output[i]
                    pred = self.student(xb)
                    tot_loss += self.loss_fn(self.student(xb), yb, tb)
                    nv = len(self.data.valid_dl)
                    print(epoch, tot_loss/nv)

        return tot_loss/nv


    def get_teacher_output(self, teacher, dl):
        teacher.eval()
        output = []
        for i, (xb, _ ) in enumerate(dl):
            xb.cpu()
            out = teacher(xb)
            output.append(out)
        
        return output