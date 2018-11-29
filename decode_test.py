from allennlp.predictors.predictor import Predictor
from nlpete.models.copynet_elmo import CopyNet
#from nlpete.data.dataset_readers import CopyNetDatasetReader

#_, loaded_model = \
 #           ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)
predictor = Predictor.from_path("./models/model.tar.gz")
output_dict = predictor.predict("these tokens should be copied over : hello world")


f = open("./data/coord/test.tsv", "r")
for line in f.readlines():
    predictor.predict(line.replace("\n", ""))
import pdb; pdb.set_trace()