import os, sys
from pydotted import pydot

# Set base project directory to current working directory
PROJECT_DIR = os.path.abspath(os.getcwd())

# Import DD helper modules
sys.path.append(PROJECT_DIR)
import dd, dd_args

# Unsure about these:
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# Load parameters
pargs = dd_args.arg_configuration_loader()

# Setup folders
#folders = dd.setupFolders(is_colab=dd.detectColab(), PROJECT_DIR=PROJECT_DIR, pargs=pargs)

# Load Models
#dd.loadModels(folders)
#dd.loadModels2(folders)

# Report System Details
#dd.systemDetails(pargs)

# Get CUDA Device
device = dd.getDevice(pargs)
import uuid
import os

if __name__=="__main__":
 
    folders = pydot(
        {
            "root_path": os.getcwd(),
            "outDirPath": '/outputs/character-portraits',
            "model_path": 'inputs/dd-models',
        }
    )

    pargs.diffusion_model = "portrait_generator_v001"
    pargs.use_secondary_model = False
    pargs.n_batches = 1
    pargs.steps = 50
    input_dir = sys.argv[1]
    for f in os.listdir(input_dir):
        data = json.load(os.path.join(input_dir, f))
        batch_name = f.replace('json','')
        folders.batch_path  = os.path.join(input_path, batch_name)
        pargs.text_prompts ={0: ["{}:1.5".format(data['description']),
         "artstation,deviantart,vray render, unreal engine, hyperrealism, photorealism,volumetric lighting:.3"] + ['style of {}'.format(a) for a in data['artists']]}

        dd.start_run(pargs=pargs, folders=folders, device=device, is_colab=dd.detectColab())


