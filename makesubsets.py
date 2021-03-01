import os
import shutil
import random
import numpy.random
import numpy as np

# The function create_subsets(small, medium) creates folders Subsets with subsets on your computer. It needs only to be
# run once(more runs will cause errors because the folders then already exists)

# The main subsets are Random, Black and White ("BW"), Bright Colors("Bright"),
# and Dull colors("Dull")
# additionaly there are subsets "Small","Medium", "Full" for each except for the
# Random, since then the original data can be used. Noth that the "Full" versions
# have different sizes.

# It is assumed that this file (and files that will use the subsets) has the path project/code/makesubsets.py (names
# may differ) and that the whole data set is in a folder named Data with path 
# project/Data

# for the medium and small versions the same validation and testset can be used,
# therefore it lies only in the small versions.

# Paths to the subsets:

###Random

##Small:
#Training:
"../Subset/RandomSmall/train"
#Validation:
"../Subset/RandomSmall/valid"
#Test:
"../Subsets/RandomSmall/test"

##Medium:
#Training:
"../Subsets/RandomMedium/train"
#Validation:
"../Subsets/RandomSmall/valid"
#Test:
"../Subsets/RandomSmall/test"

##Full:
#Training
"../Data/train"
#Validation:
"../Data/valid"
#Test:
"../Data/test"

### Bright Colors:

##Small:
#Training:
"../Subsets/BrightSmall/train"
#Validation:
"../Subsets/BrightSmall/valid"
#Test:
"../Subsets/BrightSmall/test"

##Medium:
#Training:
"../Subsets/BrightMedium/train"
#Validation:
"../Subsets/BrightSmall/valid"
#Test:
"../Subsets/BrightSmall/test"

##Full:
#Training:
"../Subsets/BrightFull/train"
#Validation:
"../Subsets/BrightFull/valid"
#Test:
"../Subsets/BrightFull/test"

# BLack/White and Dull Colors follow similary as for Bright colors, just replace
# Bright with BW and Dull 


def create_subsets(small=5, medium=50):
    # Settings for sizes:
    
    np.random.seed(5)
    species = os.listdir("../Data/train")
    n_species = len(species)
    species_per_subset = 15  # number of species in a subset

    ###
    
    os.mkdir("../Subsets") 
    
    subsets = ["RandomSmall", "RandomMedium", "BrightSmall",
               "BrightMedium", "BrightFull", "DullSmall", "DullMedium",
               "DullFull", "BWSmall", "BWMedium", "BWFull"]
    
    # create folders
    for foldername in subsets:
        path = os.path.join("../Subsets", foldername)
        os.mkdir(path)
        os.mkdir(path+"/train")
        for innerfolder in ["valid", "test"]:
            if "Medium" not in path : 
                innerpath = os.path.join(path, innerfolder) 
                os.mkdir(innerpath)
    
                
    
    def move_trainpics(subset, species, size):
        pics = os.listdir("../Data/train/"+species)
        indices = random.sample(range(0,len(pics)), size)
        selected_pics = [pics[i] for i in indices]    
        for pic in selected_pics:
            shutil.copy("../Data/train/"+species+"/"+pic,"../Subsets/"+subset+"/train/"+species)
    
    def move_imgs(selected_species, subset):
        for species in selected_species:
            os.mkdir("../Subsets/"+subset+"Small/train/"+species)
            os.mkdir("../Subsets/"+subset+"Medium/train/"+species)
          
            shutil.copytree("../Data/valid" +"/"+species,"../Subsets/"+subset+"Small"+"/valid/" +species)
            shutil.copytree("../Data/test" +"/"+species,"../Subsets/"+subset+"Small"+"/test/"+species)
            
            move_trainpics(subset+"Small", species, small)    
            move_trainpics(subset+"Medium", species, medium)
               
            
    #Random:
      
    selected_indicesR = numpy.random.randint(0,n_species, species_per_subset)
    selected_speciesR = [species[i] for i in selected_indicesR]
    move_imgs(selected_speciesR, "Random")
    shutil.copytree("../Data", "../Subsets/AllData")
    
    
    
    
    #Black and white
    BW=["ALBATROSS","AMERICAN AVOCET", "AMERICAN COOT", "ANHINGA", "BALD EAGLE", "BELTED KINGFISHER",
        "BLACK SKIMMER", "BLACK SWAN","BLACK VULTURE", "BLUE HERON","BOBOLINK", "CASPIAN TERN",
        "COCKATOO", "COMMON GRACKLE", "COMMON LOON", "CRESTED AUKLET","CROW", "EMPEROR PENGUIN",
        "EURASIAN MAGPIE","FRIGATE", "GLOSSY IBIS", "GREY PLOVER", "GUINEAFOWL", "IMPERIAL SHAQ",
        "INCA TERN", "LARK BUNTING", "MALEO", "MARABOU STORK", "MASKED BOOBY", "NORTHERN GANNET",
        "NORTHERN GOSHAWK", "OSTRICH", "PELICAN", "RAZORBILL", "RED FACED CORMORANT","RED WINGED BLACKBIRD",
        "RING-BILLED GULL", "ROUGH LEG BUZZARD", "SAND MARTIN", "SNOWY EGRET", "SNOWY OWL", 
        "SPOONBILL", "STEAMER DUCK","TIT MOUSE","TRUMPTER SWAN", "WHITE NECKED RAVEN", 
        "WHITE TAILED TROPIC", "WILD TURKEY"]
    
    #Colors Bright
    CB=["ALEXANDRINE PARAKEET","AMERICAN GOLDFINCH", "AMERICAN REDSTART","ANNAS HUMMINGBIRD",
        "ARARIPE MANAKIN", "BALTIMORE ORIOLE", "BANANAQUIT",  "BAY-BREASTED WARBLER", "BIRD OF PARADISE", 
        "BLACKBURNIAM WARBLER", "CANARY", "CAPE MAY WARBLER", "CARMINE BEE-EATER", 
        "CHARA DE COLLAR",  "COCK OF THE  ROCK", "COUCHS KINGBIRD","CROWNED PIGEON",
        "CUBAN TODY", "CURL CRESTED ARACURI", "D-ARNAUDS BARBET","EASTERN BLUEBIRD", "EASTERN MEADOWLARK",
        "EASTERN ROSELLA", "ELEGANT TROGON", "ELLIOTS  PHEASANT", "FLAME TANAGER", "FLAMINGO", "GOLDEN CHEEKED WARBLER", 
        "GOLDEN CHLOROPHONIA", "GOLDEN PHEASANT", "GOULDIAN FINCH", "GREEN JAY","HOODED MERGANSER","HOOPOES",
        "HORNBILL","HOUSE FINCH", "HOUSE SPARROW", "HYACINTH MACAW", "INDIGO BUNTING", "JAVAN MAGPIE",
        "KING VULTURE", "LILAC ROLLER", "MALLARD DUCK", "MANDRIN DUCK", "MIKADO  PHEASANT", "NICOBAR PIGEON",
        "NORTHERN CARDINAL", "NORTHERN JACANA", "NORTHERN PARULA","NORTHERN RED BISHOP", "OCELLATED TURKEY",
        "PAINTED BUNTIG", "PARADISE TANAGER","PARUS MAJOR","PEACOCK", "PINK ROBIN",
        "PURPLE FINCH", "PURPLE GALLINULE", "PURPLE SWAMPHEN","QUETZAL","RAINBOW LORIKEET", "RED FACED WARBLER",
        "RED HEADED WOODPECKER","RED HONEY CREEPER", "RED THROATED BEE EATER", "ROBIN", "ROSY FACED LOVEBIRD",
        "RUFOUS KINGFISHER", "SCARLET IBIS", "SCARLET MACAW", "SPANGLED COTINGA",
        "SPLENDID WREN", "STORK BILLED KINGFISHER", "STRAWBERRY FINCH", "TAIWAN MAGPIE", "TOUCHAN","TOWNSENDS WARBLER",
        "TREE SWALLOW", "TURQUOISE MOTMOT", "VENEZUELIAN TROUPIAL", "VERMILION FLYCATHER", 
        "WHITE CHEEKED TURACO", "WILSONS BIRD OF PARADISE","YELLOW HEADED BLACKBIRD"] 
        
    #Colors Dull
    CD=["AMERICAN BITTERN","AMERICAN KESTREL", "AMERICAN PIPIT", "BARN OWL", "BARN SWALLOW", "BAR-TAILED GODWIT",
        "BLACK FRANCOLIN", "BLACK-CAPPED CHICKADEE","CINNAMON TEAL", "BLACK-THROATED SPARROW", "BLUE GROUSE",
        "BROWN NOODY", "BROWN THRASHER", "CACTUS WREN", "CALIFORNIA GULL", "CALIFORNIA QUAIL", 
        "CHIPPING SPARROW","COMMON POORWILL", "DARK EYED JUNCO", "EMU","GILDED FLICKER", "MOURNING DOVE",
        "GRAY PARTRIDGE", "KILLDEAR", "LONG-EARED OWL","MYNA", "NORTHERN FLICKER",
        "OSPREY", "PEREGRINE FALCON", "PURPLE MARTIN", "RED HEADED DUCK", "RED WISKERED BULBUL",
        "RING-NECKED PHEASANT","ROADRUNNER", "ROCK DOVE","SHOEBILL", "SORA", "TEAL DUCK",
        "TURKEY VULTURE", "VARIED THRUSH", "WOOD DUCK"]
    
    
    def move_subset(subset, spec_list):
        selected_species = [i for i in spec_list]
        move_imgs(selected_species, subset) # all but full
        for species in spec_list:
            os.mkdir("../Subsets/"+subset+"Full/train/"+species)
            move_trainpics(subset+"Full", species, len(spec_list))
            
            shutil.copytree("../Data/valid" +"/"+species,"../Subsets/"+subset+"Full"+"/valid/" +species)
            shutil.copytree("../Data/test" +"/"+species,"../Subsets/"+subset+"Full"+"/test/" +species)
    
    move_subset("BW", BW)
    move_subset("Bright", CB)
    move_subset("Dull", CD)
        
create_subsets()

