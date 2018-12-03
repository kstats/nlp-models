#from allennlp.predictors.predictor import lo
from nlpete.predictors.copynet import CopyNetPredictor
from allennlp.models.archival import load_archive
from nlpete.training.metrics import *
from nlpete.data.dataset_readers import CopyNetDatasetReader
from allennlp.predictors import Predictor



from nlpete.models.copynet_elmo import CopyNet
#from nlpete.data.dataset_readers import CopyNetDatasetReader

#_, loaded_model = \
 #           ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)
archive = load_archive('./models/model.tar.gz')

predictor = Predictor.from_archive(archive, "copynet")

#predictor = CopyNetPredictor.from_path("./models/model.tar.gz")
#output_dict = predictor.predict("these tokens should be copied over : hello world")


f = open("./data/coord/test.tsv", "r")
f2 = open("./data/coord/model_output.txt", "w")
for line in f.readlines():
    f2.write(" ".join(predictor.predict(line.replace("\n", ""))['predicted_tokens'][0]) + "\n")
