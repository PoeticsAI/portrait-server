import os, sys

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
folders = dd.setupFolders(is_colab=dd.detectColab(), PROJECT_DIR=PROJECT_DIR, pargs=pargs)

# Load Models
dd.loadModels(folders)
dd.loadModels2(folders)

# Report System Details
#dd.systemDetails(pargs)

# Get CUDA Device
device = dd.getDevice(pargs)
from flask import jsonify
from flask import Response, Flask
from flask import make_response, request, current_app, render_template
import uuid
import os
app = Flask(__name__)
app.secret_key = "weeeeeeeeee"
@app.route("/list", methods=["GET"])
def list_results():
    return jsonify({'results': os.listdir('./images_out')})
@app.route("/portrait", methods=["POST"])
def portrait():
    pargs.diffusion_model = "portrait_generator_v001"
    pargs.use_secondary_model = False
    pargs.batch_name = "portrait-" + str(uuid.uuid1())
    pargs.n_batches = 1
    pargs.steps = 50
    pargs.text_prompts ={0: ["{}:1.5".format(request.args['description']),
         "style of Julia Condon",
         "artstation,deviantart,vray render, unreal engine, hyperrealism, photorealism,volumetric lighting:.3"]}

    dd.start_run(pargs=pargs, folders=folders, device=device, is_colab=dd.detectColab())
    print(pargs.batch_name)
    return jsonify({'batch': pargs.batch_name})

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080, debug=True, use_reloader=True)
