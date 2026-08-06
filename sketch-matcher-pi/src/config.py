import os
from pathlib import Path

# =============================================================================
# PROJECT ROOT
# All paths are relative to this file's location (src/)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# DATA PATHS
# - raw/       : original downloaded datasets
# - processed/ : preprocessed .npy files (cached after first run)
# =============================================================================
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# DATASET CONFIG
# =============================================================================
DATASETS = {
    "sketchy": {
        "url": "https://www.kaggle.com/datasets/balraj98/sketchydataset",
        "raw_dir": RAW_DIR / "sketchy",
        "categories": 125,
    },
    "quickdraw": {
        "url": "https://console.cloud.google.com/storage/browser/quickdraw-dataset/sketchrnn",
        "raw_dir": RAW_DIR / "quickdraw",
        "categories": 345,
        "use_subset": True,
        "subset_size": 50000,
    },
    "tuberlin": {
        "url": "https://www.kaggle.com/datasets/borismokeev/tuberlin-sketch-dataset",
        "raw_dir": RAW_DIR / "tuberlin",
        "categories": 250,
    },
    "imagenetsketch": {
        "url": "https://www.kaggle.com/datasets/wanghaohan/imagenetsketch",
        "raw_dir": RAW_DIR / "imagenetsketch",
        "categories": 1000,
    },
}

# Which datasets to actually USE during preprocessing/training.
# Sketchy is always required (it provides the photo side + paired sketches).
# The others only ADD sketch-side variety; their categories are mapped to
# Sketchy's canonical categories by name (see class_mapping.py).
USE_QUICKDRAW = True
USE_TUBERLIN = True
# ImageNet-Sketch ships with synset folder names (n02342885) and no mapping
# file in the Kaggle mirror, so category names cannot be mapped to Sketchy's
# canonical set. Enable only if a synset -> name file is provided.
USE_IMAGENETSKETCH = False
# Per-category caps keep total RAM sane: ~0.15 MB per 224x224x3 uint8 image.
# User trains on a machine with a 96 GB VRAM GPU -> caps are set HIGH for
# maximum data diversity:
#   QuickDraw 125 mapped cats x 1000          = ~125k extra sketches (+~19 GB)
#   ImageNet-Sketch 125 mapped cats x 200     = ~25k  extra sketches (+~4 GB)
#   TU-Berlin                                = unlimited (~250 cats)
# Lower the caps if the machine has limited system RAM (not VRAM).
QUICKDRAW_MAX_PER_CATEGORY = 1000
TUBERLIN_MAX_PER_CATEGORY = 0           # 0 = unlimited
IMAGENETSKETCH_MAX_PER_CATEGORY = 200   # huge dataset: sample evenly

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
IMG_SIZE = 224          # Input size for CNN (224x224 = standard backbone size)
IMG_CHANNELS = 3         # Convert grayscale to 3-channel for pretrained backbones
BINARY_THRESHOLD = 128   # Otsu threshold for binarization (fallback value)
PADDING = 10             # Padding pixels around cropped bounding box
NORMALIZE_MEAN = [0.485, 0.456, 0.406]   # ImageNet mean (for transfer learning)
NORMALIZE_STD = [0.229, 0.224, 0.225]    # ImageNet std (for transfer learning)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Backbone options (all pretrained on ImageNet, all TFLite-convertible):
#   "mobilenetv2"      -> 3.5M params, ~150ms on Pi 5 (older STUDENT option)
#   "mobilenetv3large" -> 5.4M params, ~100-150ms on Pi 5 (STUDENT - ships to
#                         Pi; better accuracy than V2 at similar FLOPs and has
#                         a built-in [-1,1] rescale, so [0,1] inputs are fed
#                         correctly)
#   "efficientnetv2s"  -> 21M params, high accuracy (TEACHER)
#   "convnexttiny"     -> 28M params, top accuracy (TEACHER)
BACKBONE = "mobilenetv3large"
TEACHER_BACKBONE = "convnexttiny"   # teacher for distillation (not deployed)
EMBEDDING_DIM = 256          # 128 = light, 256 = balanced, 512 = max capacity
DROPOUT_RATE = 0.3
USE_PRETRAINED = True        # Start from ImageNet weights

# =============================================================================
# ARC FACE HEAD (additive angular margin classifier)
# =============================================================================
# Adds a classification head on the L2-normed embedding (Deng et al. 2019).
# ArcFace is a proven accuracy booster for embedding retrieval (the sketch and
# photo branches each get a shared classifier). The embedding stays
# L2-normalized for retrieval, so TFLite export is unaffected.
# Generator yields category labels as extra model inputs AND CE targets.
USE_ARC_FACE = True
ARC_FACE_MARGIN = 0.5         # additive angular margin (radians)
ARC_FACE_SCALE = 64.0         # logit scale s
ARC_FACE_LAMBDA = 0.5         # CE loss weight per branch (pair loss weight = 1.0)
NUM_CLASSES = 125             # must match preprocessed category count

# Knowledge distillation (teacher -> student for the Pi)
# If False, the BACKBONE model is trained directly (no distillation).
ENABLE_DISTILLATION = True
DISTILL_ALPHA = 0.5          # weight of distillation loss vs contrastive loss

# Transfer learning freeze schedule
FREEZE_BACKBONE_STAGE1 = True    # Stage 1: freeze all backbone layers
UNFREEZE_FROM_STAGE2 = -12       # Stage 2: unfreeze last 12 backbone layers
FULL_FINETUNE_STAGE3 = True      # Stage 3: unfreeze everything

# =============================================================================
# LOSS FUNCTION
# =============================================================================
# Options: "contrastive", "circle"
#   contrastive -> classic (1-Y)*0.5*D^2 + Y*0.5*max(0, M-D)^2
#   circle      -> SOTA metric loss (Sun et al. 2020), adaptive weights,
#                  better convergence on hard pairs (recommended)
LOSS_TYPE = "circle"
CONTRASTIVE_MARGIN = 1.0     # Contrastive loss margin
CIRCLE_M = 0.25              # Circle loss margin
CIRCLE_GAMMA = 80            # Circle loss gamma (scale factor)
SIM_TEMPERATURE = 0.1        # temperature for cosine similarity (sharpening)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
TRAIN_SPLIT = 0.80          # 80% for training
VAL_SPLIT = 0.10            # 10% for validation (threshold tuning)
TEST_SPLIT = 0.10           # 10% for final evaluation

# Stage 1: train dense layers only (frozen backbone)
STAGE1_EPOCHS = 100
# Batch sizes are overridable via env (HPC/MIG sizing) -- e.g. in a PBS script:
#   export SKETCH_BATCH1=64 SKETCH_BATCH2=64 SKETCH_BATCH3=32
STAGE1_BATCH_SIZE = int(os.environ.get("SKETCH_BATCH1", "256"))
STAGE1_LEARNING_RATE = 0.001

# Stage 2: unfreeze last 12 backbone layers
STAGE2_EPOCHS = 60
STAGE2_BATCH_SIZE = int(os.environ.get("SKETCH_BATCH2", "128"))
STAGE2_LEARNING_RATE = 0.0001

# Stage 3: full fine-tune
STAGE3_EPOCHS = 150
STAGE3_BATCH_SIZE = int(os.environ.get("SKETCH_BATCH3", "64"))
STAGE3_LEARNING_RATE = 0.00001

# Early stopping
EARLY_STOPPING_PATIENCE = 15   # More patience for longer training
REDUCE_LR_PATIENCE = 5         # Reduce LR if no improvement for 5 epochs
REDUCE_LR_FACTOR = 0.5         # Multiply LR by this factor

# =============================================================================
# PAIR GENERATION
# =============================================================================
PAIRS_PER_EPOCH = 80000        # fresh random pairs every epoch
POSITIVE_PAIRS_PER_CATEGORY = 500    # diversity floor per category
NEGATIVE_PAIRS_PER_EPOCH = 80000
NEGATIVE_TO_POSITIVE_RATIO = 1.0     # 1:1 ratio (balanced)

# Hard-negative sampling: with this probability, a negative pair's photo
# category is drawn from a "confusable" list (cat<->dog, car<->truck, ...)
# instead of a fully random category. Forces the model to discriminate
# between visually similar categories -> much lower false matches.
HARD_NEGATIVE_RATIO = 0.5
CONFUSABLE_PAIRS = {
    "cat": ["dog", "fox", "lion", "tiger"],
    "dog": ["cat", "fox", "wolf"],
    "car": ["truck", "bus", "van", "pickup"],
    "truck": ["car", "bus"],
    "bus": ["car", "truck", "train"],
    "apple": ["pear", "orange", "watermelon"],
    "pear": ["apple", "banana"],
    "cup": ["mug", "bowl", "bottle"],
    "mug": ["cup", "bowl"],
    "bottle": ["cup", "wine_bottle"],
    "chair": ["sofa", "bench", "stool"],
    "sofa": ["chair", "bench", "couch"],
    "bench": ["chair", "sofa"],
    "bicycle": ["motorcycle", "scooter"],
    "motorcycle": ["bicycle", "scooter"],
    "knife": ["fork", "scissors", "sword"],
    "fork": ["knife", "spoon"],
    "fish": ["whale", "dolphin", "shark"],
    "whale": ["fish", "dolphin"],
    "bird": ["airplane", "butterfly", "duck"],
    "duck": ["bird", "goose"],
    "elephant": ["rhino", "hippo"],
    "rhino": ["elephant", "rhinoceros"],
    "circle": ["star", "moon", "sun"],
    "star": ["circle", "cross"],
    "mouse": ["rat", "hamster"],
    "sheep": ["goat", "cow"],
    "cow": ["sheep", "goat", "buffalo"],
    "clock": ["watch", "compass"],
    "watch": ["clock"],
    "hammer": ["axe", "mallet"],
    "shoes": ["slippers", "boots"],
    "shoe": ["slipper", "boot"],
    "umbrella": ["parachute"],
    "piano": ["guitar", "keyboard"],
    "guitar": ["piano", "violin", "ukulele"],
    "helmet": ["hat", "cap"],
    "hat": ["helmet", "cap"],
    "airplane": ["bird", "helicopter", "rocket"],
    "helicopter": ["airplane", "bird"],
    "rocket": ["airplane", "helicopter"],
    "lamp": ["candle", "flashlight"],
    "candle": ["lamp", "torch"],
    "sailboat": ["boat", "ship"],
    "boat": ["sailboat", "ship", "canoe"],
    "train": ["bus", "truck", "subway"],
    "tank": ["car", "truck"],
    "forklift": ["truck", "car"],
    "camera": ["binoculars", "phone", "telescope"],
    "phone": ["camera", "tablet"],
    "laptop": ["computer", "tablet", "keyboard"],
    "keyboard": ["laptop", "piano", "computer"],
    "mouse": ["rat", "hamster"],
    "spider": ["crab", "insect", "bug"],
    "ant": ["spider", "bug", "bee"],
    "bee": ["ant", "wasp", "fly", "butterfly"],
    "butterfly": ["bee", "moth", "bird"],
    "flower": ["tree", "plant", "sunflower"],
    "tree": ["flower", "plant", "bush"],
    "leaf": ["flower", "plant", "tree"],
    "mushroom": ["umbrella", "flower"],
    "pineapple": ["pinecone", "corn"],
    "corn": ["pineapple", "banana"],
    "banana": ["corn", "pineapple", "apple"],
    "onion": ["apple", "orange", "potato"],
    "potato": ["onion", "apple"],
    "tomato": ["apple", "orange", "cherry"],
    "cherry": ["apple", "tomato"],
    "pepper": ["carrot", "chili"],
    "carrot": ["pepper", "radish"],
    "egg": ["apple", "ball"],
    "ball": ["egg", "apple", "circle"],
    "snowman": ["person", "ghost"],
    "person": ["snowman", "ghost", "man"],
    "face": ["person", "head"],
    "eye": ["nose", "ear", "head"],
    "nose": ["eye", "ear", "head"],
    "ear": ["eye", "nose", "head"],
    "head": ["face", "eye", "nose", "ear"],
    "hand": ["foot", "paw", "glove"],
    "foot": ["hand", "paw", "shoe"],
    "paw": ["hand", "foot", "glove"],
    "glove": ["hand", "mitten"],
    "mitten": ["glove", "hand"],
    "sock": ["shoe", "foot", "slipper"],
    "book": ["notebook", "magazine", "newspaper"],
    "newspaper": ["book", "magazine", "paper"],
    "paper": ["book", "newspaper", "tissue"],
    "tissue": ["paper", "napkin", "toilet"],
    "toilet": ["tissue", "sink", "bathtub"],
    "bathtub": ["toilet", "sink", "pool"],
    "sink": ["toilet", "bathtub", "fountain"],
    "fountain": ["sink", "pool", "bathtub"],
    "pool": ["fountain", "bathtub", "lake"],
    "lake": ["pool", "ocean", "river"],
    "river": ["lake", "ocean", "stream"],
    "ocean": ["lake", "river", "pool"],
    "wave": ["ocean", "river", "water"],
    "cloud": ["smoke", "fog", "snow"],
    "snow": ["cloud", "ice", "rain"],
    "rain": ["snow", "cloud", "water"],
    "ice": ["snow", "water", "glass"],
    "water": ["ice", "rain", "river", "lake", "ocean"],
    "lightning": ["thunder", "cloud", "spark"],
    "sun": ["moon", "star", "circle", "light"],
    "moon": ["sun", "star", "crescent"],
    "crescent": ["moon", "star"],
    "key": ["lock", "keyhole"],
    "lock": ["key", "padlock"],
    "sword": ["knife", "dagger", "stick"],
    "dagger": ["sword", "knife"],
    "stick": ["sword", "baton", "wand"],
    "wand": ["stick", "baton"],
    "pencil": ["pen", "brush", "crayon", "marker"],
    "pen": ["pencil", "marker", "crayon", "brush"],
    "brush": ["pencil", "pen", "paintbrush"],
    "paintbrush": ["brush", "pencil", "pen"],
    "crayon": ["pencil", "pen", "marker"],
    "marker": ["pen", "pencil", "crayon"],
    "scissors": ["knife", "fork", "pliers"],
    "pliers": ["scissors", "tweezers", "hammer"],
    "tweezers": ["pliers", "scissors"],
    "axe": ["hammer", "hatchet"],
    "hatchet": ["axe", "hammer"],
    "saw": ["knife", "axe", "sword"],
    "drill": ["screwdriver", "hammer"],
    "screwdriver": ["drill", "wrench"],
    "wrench": ["screwdriver", "pliers"],
    "spoon": ["fork", "knife", "ladle"],
    "ladle": ["spoon", "scoop"],
    "scoop": ["ladle", "spoon", "shovel"],
    "shovel": ["scoop", "spade", "hoe"],
    "spade": ["shovel", "hoe", "pick"],
    "hoe": ["shovel", "spade", "rake"],
    "rake": ["hoe", "shovel", "broom"],
    "broom": ["rake", "mop", "brush"],
    "mop": ["broom", "brush"],
    "bucket": ["bowl", "pot", "bin"],
    "pot": ["bucket", "pan", "bowl"],
    "pan": ["pot", "frying_pan", "bowl"],
    "frying_pan": ["pan", "pot"],
    "bowl": ["cup", "mug", "bucket", "pot"],
    "plate": ["bowl", "saucer", "tray"],
    "saucer": ["plate", "cup", "bowl"],
    "tray": ["plate", "saucer", "platter"],
    "cupcake": ["cake", "muffin", "donut"],
    "muffin": ["cupcake", "cake", "donut"],
    "donut": ["bagel", "cake", "cookie"],
    "bagel": ["donut", "bread", "roll"],
    "bread": ["bagel", "roll", "toast"],
    "toast": ["bread", "bagel"],
    "roll": ["bagel", "bread", "bun"],
    "bun": ["roll", "bread", "bagel"],
    "cake": ["pie", "cupcake", "cookie"],
    "pie": ["cake", "tart", "pizza"],
    "tart": ["pie", "cake"],
    "pizza": ["pie", "cake", "flatbread"],
    "cookie": ["cake", "donut", "biscuit"],
    "biscuit": ["cookie", "cracker", "bread"],
    "cracker": ["biscuit", "cookie"],
    "candy": ["lollipop", "chocolate", "gum"],
    "lollipop": ["candy", "lollipop", "sucker"],
    "chocolate": ["candy", "brownie"],
    "brownie": ["chocolate", "cake", "cookie"],
    "ice_cream": ["cone", "sundae", "yogurt", "popsicle"],
    "popsicle": ["ice_cream", "lollipop"],
    "sundae": ["ice_cream", "cone", "parfait"],
    "cone": ["ice_cream", "sundae", "traffic_cone"],
    "yogurt": ["ice_cream", "sundae", "parfait"],
    "parfait": ["sundae", "yogurt", "ice_cream"],
    "fries": ["french_fries", "chips", "fries"],
    "french_fries": ["fries", "chips", "hash_browns"],
    "chips": ["fries", "french_fries", "crackers"],
    "hash_browns": ["fries", "french_fries", "potato"],
    "sandwich": ["burger", "wrap", "hotdog"],
    "burger": ["sandwich", "hotdog", "wrap"],
    "hotdog": ["sandwich", "burger", "sausage"],
    "sausage": ["hotdog", "hot_dog", "brat"],
    "hot_dog": ["hotdog", "sausage", "sandwich"],
    "wrap": ["sandwich", "burrito", "taco"],
    "burrito": ["wrap", "taco", "sandwich"],
    "taco": ["burrito", "wrap", "tortilla"],
    "tortilla": ["taco", "burrito", "wrap"],
    "sushi": ["rice", "roll", "tempura"],
    "rice": ["sushi", "grains", "corn"],
    "ramen": ["noodles", "pasta", "soup"],
    "noodles": ["ramen", "pasta", "spaghetti"],
    "pasta": ["spaghetti", "noodles", "macaroni"],
    "spaghetti": ["pasta", "noodles", "ramen"],
    "soup": ["ramen", "stew", "broth"],
    "stew": ["soup", "chili"],
    "salad": ["lettuce", "greens", "spinach"],
    "lettuce": ["salad", "cabbage", "spinach"],
    "cabbage": ["lettuce", "broccoli"],
    "spinach": ["lettuce", "salad", "greens"],
    "broccoli": ["cauliflower", "cabbage", "lettuce"],
    "cauliflower": ["broccoli", "cabbage"],
    "zucchini": ["cucumber", "squash", "eggplant"],
    "cucumber": ["zucchini", "pickle", "squash"],
    "pickle": ["cucumber", "zucchini"],
    "squash": ["zucchini", "pumpkin", "gourd"],
    "pumpkin": ["squash", "gourd", "orange"],
    "gourd": ["pumpkin", "squash"],
    "eggplant": ["zucchini", "cucumber"],
    "asparagus": ["broccoli", "leek", "celery"],
    "leek": ["asparagus", "green_onion", "onion"],
    "celery": ["asparagus", "leek"],
    "green_onion": ["leek", "onion", "scallion"],
    "scallion": ["green_onion", "leek", "onion"],
    "grape": ["berry", "raisin", "olive"],
    "berry": ["grape", "raspberry", "strawberry"],
    "raspberry": ["berry", "strawberry", "grape"],
    "strawberry": ["raspberry", "berry"],
    "blueberry": ["berry", "grape", "raisin"],
    "raisin": ["grape", "blueberry"],
    "olive": ["grape", "berry"],
    "peach": ["pear", "apple", "apricot"],
    "apricot": ["peach", "pear", "plum"],
    "plum": ["apricot", "peach", "pear"],
    "kiwi": ["pear", "lemon", "lime"],
    "lemon": ["lime", "orange", "grapefruit"],
    "lime": ["lemon", "green_apple", "kiwi"],
    "grapefruit": ["lemon", "orange", "lime"],
    "orange": ["lemon", "grapefruit", "apple"],
    "green_apple": ["apple", "lime", "pear"],
    "avocado": ["pear", "green_apple", "kiwi"],
    "coconut": ["hairy_ball", "round_ball", "orange"],
    "hairy_ball": ["coconut", "pom_pom"],
    "pom_pom": ["hairy_ball", "cotton"],
    "cotton": ["pom_pom", "cloud", "snow"],
    "snowflake": ["snow", "star", "ice"],
    "crystal": ["ice", "snowflake", "gem"],
    "gem": ["crystal", "diamond", "jewel"],
    "diamond": ["gem", "crystal", "rhombus"],
    "rhombus": ["diamond", "square", "kite"],
    "kite": ["rhombus", "diamond", "bird"],
    "square": ["rectangle", "rhombus", "box"],
    "rectangle": ["square", "rhombus", "box"],
    "box": ["square", "rectangle", "crate"],
    "crate": ["box", "cage", "container"],
    "cage": ["crate", "jail", "birdcage"],
    "jail": ["cage", "prison", "cell"],
    "cell": ["jail", "cage", "phone_cell"],
    "prison": ["jail", "cage"],
    "birdcage": ["cage", "bird", "crate"],
    "container": ["crate", "box", "bin"],
    "bin": ["container", "bucket", "trash"],
    "trash": ["bin", "dumpster", "waste"],
    "dumpster": ["trash", "bin", "container"],
    "waste": ["trash", "dumpster", "recycling"],
    "recycling": ["waste", "trash", "bin"],
    "shopping_cart": ["basket", "cart", "stroller"],
    "basket": ["shopping_cart", "cart", "hamper"],
    "cart": ["shopping_cart", "basket", "wagon"],
    "wagon": ["cart", "stroller", "shopping_cart"],
    "stroller": ["wagon", "baby", "carriage"],
    "carriage": ["stroller", "wagon", "chariot"],
    "chariot": ["carriage", "wagon"],
    "baby": ["stroller", "doll", "infant"],
    "doll": ["baby", "action_figure", "puppet"],
    "puppet": ["doll", "marionette", "action_figure"],
    "action_figure": ["doll", "puppet", "toy"],
    "toy": ["doll", "action_figure", "teddy_bear", "ball"],
    "teddy_bear": ["toy", "bear", "stuffed_animal"],
    "stuffed_animal": ["teddy_bear", "toy", "plush"],
    "plush": ["stuffed_animal", "teddy_bear", "pillow"],
    "pillow": ["plush", "cushion", "blanket"],
    "cushion": ["pillow", "seat", "mattress"],
    "mattress": ["pillow", "cushion", "bed"],
    "blanket": ["pillow", "towel", "sheet"],
    "sheet": ["blanket", "towel", "bed"],
    "towel": ["blanket", "sheet", "rag"],
    "rag": ["towel", "cloth", "mop"],
    "cloth": ["rag", "towel", "fabric"],
    "fabric": ["cloth", "linen", "towel"],
    "linen": ["fabric", "cloth", "sheet"],
    "curtain": ["sheet", "drape", "blind"],
    "drape": ["curtain", "blind", "sheet"],
    "blind": ["curtain", "drape", "shade"],
    "shade": ["blind", "curtain", "umbrella"],
    "rug": ["carpet", "mat", "tapestry"],
    "carpet": ["rug", "mat", "tapestry"],
    "mat": ["rug", "carpet", "doormat"],
    "doormat": ["mat", "rug", "welcome_mat"],
    "tapestry": ["rug", "carpet", "wall_hanging"],
    "wall_hanging": ["tapestry", "painting", "poster"],
    "painting": ["picture", "portrait", "canvas"],
    "picture": ["painting", "photo", "portrait"],
    "photo": ["picture", "painting", "selfie"],
    "portrait": ["painting", "picture", "photo"],
    "canvas": ["painting", "tent", "tarpaulin"],
    "tent": ["canvas", "teepee", "awning"],
    "teepee": ["tent", "wigwam"],
    "wigwam": ["teepee", "tent"],
    "awning": ["tent", "canopy", "shade"],
    "canopy": ["awning", "tent", "umbrella"],
    "statue": ["sculpture", "bust", "monument"],
    "sculpture": ["statue", "bust", "monument"],
    "bust": ["statue", "sculpture", "head"],
    "monument": ["statue", "sculpture", "obelisk"],
    "obelisk": ["monument", "spire", "pyramid"],
    "pyramid": ["obelisk", "triangle", "tent"],
    "triangle": ["pyramid", "cone", "arrow"],
    "cone": ["triangle", "ice_cream", "pyramid"],
    "arrow": ["triangle", "dart", "pointer"],
    "dart": ["arrow", "needle", "dartboard"],
    "dartboard": ["dart", "target", "bullseye"],
    "target": ["dartboard", "bullseye", "circle"],
    "bullseye": ["target", "dartboard", "circle"],
    "spire": ["obelisk", "church", "tower"],
    "tower": ["spire", "church", "skyscraper"],
    "skyscraper": ["tower", "building", "church"],
    "building": ["skyscraper", "tower", "house", "apartment"],
    "apartment": ["building", "house", "hotel"],
    "hotel": ["apartment", "building", "inn"],
    "inn": ["hotel", "motel", "tavern"],
    "motel": ["hotel", "inn"],
    "tavern": ["inn", "pub", "restaurant"],
    "restaurant": ["tavern", "cafe", "diner"],
    "cafe": ["restaurant", "diner", "coffee_shop"],
    "diner": ["cafe", "restaurant"],
    "coffee_shop": ["cafe", "diner", "bistro"],
    "bistro": ["cafe", "coffee_shop"],
    "house": ["building", "apartment", "home", "cottage"],
    "cottage": ["house", "cabin", "hut"],
    "cabin": ["cottage", "hut", "log_cabin"],
    "hut": ["cabin", "cottage", "shed"],
    "log_cabin": ["cabin", "hut"],
    "shed": ["hut", "barn", "garage"],
    "barn": ["shed", "stable", "garage"],
    "garage": ["barn", "shed", "car"],
    "stable": ["barn", "corral", "ranch"],
    "corral": ["stable", "pen", "fence"],
    "pen": ["corral", "cage", "enclosure"],
    "fence": ["corral", "gate", "railing"],
    "gate": ["fence", "door", "portal"],
    "portal": ["gate", "door", "window"],
    "door": ["gate", "portal", "entrance"],
    "entrance": ["door", "gate", "archway"],
    "archway": ["entrance", "bridge", "tunnel"],
    "tunnel": ["archway", "bridge", "cave"],
    "cave": ["tunnel", "cavern", "grotto"],
    "cavern": ["cave", "tunnel"],
    "grotto": ["cave", "cavern"],
    "bridge": ["archway", "overpass", "viaduct"],
    "overpass": ["bridge", "viaduct", "highway"],
    "viaduct": ["bridge", "overpass"],
    "highway": ["overpass", "road", "street"],
    "road": ["highway", "street", "path"],
    "street": ["road", "highway", "avenue"],
    "avenue": ["street", "road", "boulevard"],
    "boulevard": ["avenue", "street"],
    "path": ["road", "trail", "sidewalk"],
    "trail": ["path", "road", "track"],
    "sidewalk": ["path", "trail", "street"],
    "track": ["trail", "race_track", "rail"],
    "race_track": ["track", "oval", "stadium"],
    "stadium": ["race_track", "arena", "field"],
    "arena": ["stadium", "colosseum", "field"],
    "colosseum": ["arena", "stadium"],
    "field": ["stadium", "meadow", "park"],
    "meadow": ["field", "park", "lawn"],
    "park": ["meadow", "field", "garden"],
    "lawn": ["meadow", "park", "garden"],
    "garden": ["park", "lawn", "flower_bed"],
    "flower_bed": ["garden", "flower", "planter"],
    "planter": ["flower_bed", "pot", "garden"],
    "crop": ["field", "farm", "plantation"],
    "farm": ["crop", "ranch", "field"],
    "plantation": ["crop", "farm", "field"],
    "ranch": ["farm", "stable", "corral"],
}

# =============================================================================
# EVALUATION
# =============================================================================
TOP_K = [1, 3, 5]           # Top-K accuracy metrics
DISTANCE_METRIC = "cosine"   # Options: "cosine", "euclidean"

# =============================================================================
# TFLITE EXPORT
# =============================================================================
TFLITE_FILENAME = "sketch_matcher.tflite"
TFLITE_QUANTIZE = True       # Apply int8 quantization (smaller, faster on Pi)
PHOTO_EMBEDDINGS_FILENAME = "photo_embeddings.npy"
PHOTO_LABELS_FILENAME = "photo_labels.npy"
LABELS_FILENAME = "labels.json"

# =============================================================================
# PI DEPLOYMENT
# =============================================================================
# Decision thresholds (tuned from evaluation threshold sweep)
PI_CONFIDENCE_THRESHOLD = 0.75    # Green LED if top-1 similarity > 75%
PI_REJECT_THRESHOLD = 0.45        # Below this, show "NOT FOUND" instead of a
                                  # false confident answer (open-set rejection)
PI_SHOW_TOP_K = 3                 # Show top 3 results on screen
PI_CAMERA_RESOLUTION = (1920, 1080)
PI_CAMERA_FRAMERATE = 30
PI_GPIO_BUTTON_PIN = 17           # GPIO pin for capture button
PI_GPIO_LED_GREEN = 27            # GPIO pin for green LED
PI_GPIO_LED_RED = 22              # GPIO pin for red LED

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"
TEACHER_MODEL_PATH = MODELS_DIR / "teacher_model.keras"
