import os
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# make dictionary
train_dict = {str(k): db_dict[str(idx)] for k, idx in enumerate(train_index)}

#
with open(os.path.join(db_dir, "train_dict.json"), "w") as file:
            json.dump(train_dict, file, indent=4, cls=NumpyEncoder, sort_keys=True)