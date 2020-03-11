class DistillationLearner():
    def __init__(self, teacher, student, data, loss_fn, opt):
        self.teacher = teacher
        self.student = student
        self.data = data
        self.loss_fn = loss_fn
        self.opt = opt

    def fit(self, epochs):
        for epoch in range(epochs):
            self.student.train()
            self.teacher.eval()
            for xb, yb in self.data.train_dl:
                teacher_output = self.teacher(yb) # move to cpu

                loss = self.loss_fn(self.teacher(xb), yb, teacher_output)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            with torch.no_grad():

    def get_teacher_outputs(self, teacher, dataloader):
        teacher.eval()
        outputs = []
        




