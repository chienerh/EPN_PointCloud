# GLOBAL
EXP_NAME = 'epn_netvlad_seq567_sampling_ratio_06'
MODEL = 'epn_netvlad'
# testing: 'epn_netvlad', 'epn_gcn_netvlad', 'epn_ca_netvlad', 'epn_transformer_netvlad'
# archived: 'pointnetepn_netvlad', 'pointnetvlad_epnnetvlad' 

# DATA LOADER
DATASET = 'Oxford' # 'KITTI', 'ModelNet40'
DATASET_FOLDER = '/home/cel/data/benchmark_datasets/'
NUM_POINTS = 4096 # 4096
BATCH_NUM_QUERIES = 1
TRAIN_POSITIVES_PER_QUERY = 1
TRAIN_NEGATIVES_PER_QUERY = 1
NORMAL = False

# TRAIN
MAX_EPOCH = 30
BASE_LEARNING_RATE = 0.00005 #0.000005
MOMENTUM = 0.9
OPTIMIZER = 'adam'
DECAY_STEP = 200000
DECAY_RATE = 0.7

# LOSS
LOSS_FUNCTION = 'quadruplet' # 'quadruplet'
MARGIN_1 = 0.5
MARGIN_2 = 0.2
LOSS_LAZY = True
LOSS_IGNORE_ZERO_BATCH = False
TRIPLET_USE_BEST_POSITIVES = False

# NETWORK
FEATURE_OUTPUT_DIM = 256 #128 # global descriptor
LOCAL_FEATURE_DIM = 1024
NUM_SELECTED_POINTS = 256 #300 #1024 #256 

# SAVE PATH
LOG_DIR = 'log/'
MODEL_FILENAME = "model.ckpt"


# OTHERS
GPU = '0'

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99

RESUME = False

# TRAIN_FILE = '/home/cel/data/benchmark_datasets/training_queries_baseline.pickle'
# TEST_FILE = '/home/cel/data/benchmark_datasets/test_queries_baseline.pickle'
TRAIN_FILE = '/home/cel/data/benchmark_datasets/training_queries_baseline_seq567.pickle'
TEST_FILE = '/home/cel/data/benchmark_datasets/test_queries_baseline_seq567.pickle'
# TRAIN_FILE = '/home/cel/data/benchmark_datasets/training_queries_19overlap_seq5.pickle'
# TEST_FILE = '/home/cel/data/benchmark_datasets/test_queries_19overlap_seq5.pickle'
# TRAIN_FILE = '/home/cel/data/benchmark_datasets/training_queries_rot.pickle'
# TEST_FILE = '/home/cel/data/benchmark_datasets/test_queries_rot.pickle'

EVAL_MODEL = 'vn_dgcnn_netvlad'
RESULTS_FOLDER = 'results/pr_evaluation_vn_dgcnn_netvlad'
OUTPUT_FILE = RESULTS_FOLDER+'/results.txt'
RESUME_FILENAME = 'pretrained_models/dgcnn_related/vn_dgcnn_netvlad.ckpt'



EVAL_BATCH_SIZE = 1
EVAL_POSITIVES_PER_QUERY = 2
EVAL_NEGATIVES_PER_QUERY = 6

# EVAL_DATABASE_FILE = '/home/cel/data/benchmark_datasets/oxford_evaluation_database.pickle'
# EVAL_QUERY_FILE = '/home/cel/data/benchmark_datasets/oxford_evaluation_query.pickle'

EVAL_DATABASE_FILE = '/home/cel/data/benchmark_datasets/oxford_evaluation_database_tran.pickle'
EVAL_QUERY_FILE = '/home/cel/data/benchmark_datasets/oxford_evaluation_query_tran.pickle'

# VNN
N_KNN = 20
POOLING = 'mean'
CLUSTER_SIZE = 64


def cfg_str():
    out_string = ""
    for name in globals():
        if not name.startswith("__") and not name.__contains__("cfg_str"):
            #print(name, "=", globals()[name])
            out_string = out_string + "cfg." + name + \
                "=" + str(globals()[name]) + "\n"
    return out_string
