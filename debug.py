from options.train_options import TrainOptions
from options.resume_options import ResumeOptions

parser = TrainOptions()
# parser = ResumeOptions()
opt = parser.parse()
