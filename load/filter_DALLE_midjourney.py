from datasets import load_dataset
from tqdm import tqdm
import json

dreamfusion_prompts = [
    "a 20-sided die made out of glass",
    "a bald eagle carved out of wood",
    "a banana peeling itself",
    "a beagle in a detective's outfit",
    "a beautiful dress made out of fruit, on a mannequin. Studio lighting, high quality, high resolution",
    "a beautiful dress made out of garbage bags, on a mannequin. Studio lighting, high quality, high resolution",
    "a beautiful rainbow fish",
    "a bichon frise wearing academic regalia",
    "a blue motorcycle",
    "a blue poison-dart frog sitting on a water lily",
    "a brightly colored mushroom growing on a log",
    "a bumblebee sitting on a pink flower",
    "a bunch of colorful marbles spilling out of a red velvet bag",
    "a capybara wearing a top hat, low poly",
    "a cat with a mullet",
    "a ceramic lion",
    "a ceramic upside down yellow octopus holding a blue green ceramic cup",
    "a chihuahua wearing a tutu",
    "a chimpanzee holding a peeled banana",
    "a chimpanzee looking through a telescope",
    "a chimpanzee stirring a bubbling purple potion in a cauldron",
    "a chimpanzee with a big grin",
    "a completely destroyed car",
    "a confused beagle sitting at a desk working on homework",
    "a corgi taking a selfie",
    "a crab, low poly",
    "a crocodile playing a drum set",
    "a cute steampunk elephant",
    "a dachsund dressed up in a hotdog costume",
    "a delicious hamburger",
    "a dragon-cat hybrid",
    "a DSLR photo of a baby dragon drinking boba",
    "a DSLR photo of a baby dragon hatching out of a stone egg",
    "a DSLR photo of a baby grand piano viewed from far away",
    "a DSLR photo of a bagel filled with cream cheese and lox",
    "a DSLR photo of a bald eagle",
    "a DSLR photo of a barbecue grill cooking sausages and burger patties",
    "a DSLR photo of a basil plant",
    "a DSLR photo of a bear dancing ballet",
    "a DSLR photo of a bear dressed as a lumberjack",
    "a DSLR photo of a bear dressed in medieval armor",
    "a DSLR photo of a beautiful violin sitting flat on a table",
    "a DSLR photo of a blue jay standing on a large basket of rainbow macarons",
    "a DSLR photo of a bulldozer clearing away a pile of snow",
    "a DSLR photo of a bulldozer",
    "a DSLR photo of a cake covered in colorful frosting with a slice being taken out, high resolution",
    "a DSLR photo of a candelabra with many candles on a red velvet tablecloth",
    "a DSLR photo of a car made out of cheese",
    "a DSLR photo of A car made out of sushi",
    "a DSLR photo of a car made out pizza",
    "a DSLR photo of a cat lying on its side batting at a ball of yarn",
    "a DSLR photo of a cat magician making a white dove appear",
    "a DSLR photo of a cat wearing a bee costume",
    "a DSLR photo of a cat wearing a lion costume",
    "a DSLR photo of a cauldron full of gold coins",
    "a DSLR photo of a chimpanzee dressed like Henry VIII king of England",
    "a DSLR photo of a chimpanzee dressed like Napoleon Bonaparte",
    "a DSLR photo of a chow chow puppy",
    "a DSLR photo of a Christmas tree with donuts as decorations",
    "a DSLR photo of a chrome-plated duck with a golden beak arguing with an angry turtle in a forest",
    "a DSLR photo of a classic Packard car",
    "a DSLR photo of a cocker spaniel wearing a crown",
    "a DSLR photo of a corgi lying on its back with its tongue lolling out",
    "a DSLR photo of a corgi puppy",
    "a DSLR photo of a corgi sneezing",
    "a DSLR photo of a corgi standing up drinking boba",
    "a DSLR photo of a corgi taking a selfie",
    "a DSLR photo of a corgi wearing a beret and holding a baguette, standing up on two hind legs",
    "a DSLR photo of a covered wagon",
    "a DSLR photo of a cracked egg with the yolk spilling out on a wooden table",
    "a DSLR photo of a cup full of pens and pencils",
    "a DSLR photo of a dalmation wearing a fireman's hat",
    "a DSLR photo of a delicious chocolate brownie dessert with ice cream on the side",
    "a DSLR photo of a delicious croissant",
    "a DSLR photo of A DMC Delorean car",
    "a DSLR photo of a dog made out of salad",
    "a DSLR photo of a drum set made of cheese",
    "a DSLR photo of a drying rack covered in clothes",
    "a DSLR photo of aerial view of a ruined castle",
    "a DSLR photo of a football helmet",
    "a DSLR photo of a fox holding a videogame controller",
    "a DSLR photo of a fox taking a photograph using a DSLR",
    "a DSLR photo of a frazer nash super sport car",
    "a DSLR photo of a frog wearing a sweater",
    "a DSLR photo of a ghost eating a hamburger",
    "a DSLR photo of a giant worm emerging from the sand in the middle of the desert",
    "a DSLR photo of a goose made out of gold",
    "a DSLR photo of a green monster truck",
    "a DSLR photo of a group of dogs eating pizza",
    "a DSLR photo of a group of dogs playing poker",
    "a DSLR photo of a gummy bear playing the saxophone",
    "a DSLR photo of a hippo wearing a sweater",
    "a DSLR photo of a humanoid robot holding a human brain",
    "a DSLR photo of a humanoid robot playing solitaire",
    "a DSLR photo of a humanoid robot playing the cello",
    "a DSLR photo of a humanoid robot using a laptop",
    "a DSLR photo of a humanoid robot using a rolling pin to roll out dough",
    "a DSLR photo of a human skull",
    "a DSLR photo of a kitten standing on top of a giant tortoise",
    "a DSLR photo of a knight chopping wood",
    "a DSLR photo of a knight holding a lance and sitting on an armored horse",
    "a DSLR photo of a koala wearing a party hat and blowing out birthday candles on a cake",
    "a DSLR photo of a lemur taking notes in a journal",
    "a DSLR photo of a lion reading the newspaper",
    "a DSLR photo of a mandarin duck swimming in a pond",
    "a DSLR photo of a model of the eiffel tower made out of toothpicks",
    "a DSLR photo of a mouse playing the tuba",
    "a DSLR photo of a mug of hot chocolate with whipped cream and marshmallows",
    "a DSLR photo of an adorable piglet in a field",
    "a DSLR photo of an airplane taking off from the runway",
    "a DSLR photo of an astronaut standing on the surface of mars",
    "a DSLR photo of an eggshell broken in two with an adorable chick standing next to it",
    "a DSLR photo of an elephant skull",
    "a DSLR photo of an exercise bike in a well lit room",
    "a DSLR photo of an extravagant mansion, aerial view",
    "a DSLR photo of an ice cream sundae",
    "a DSLR photo of an iguana holding a balloon",
    "a DSLR photo of an intricate and complex dish from a michelin star restaurant",
    "a DSLR photo of An iridescent steampunk patterned millipede with bison horns",
    "a DSLR photo of an octopus playing the piano",
    "a DSLR photo of an old car overgrown by vines and weeds",
    "a DSLR photo of an old vintage car",
    "a DSLR photo of an orangutan making a clay bowl on a throwing wheel",
    "a DSLR photo of an orc forging a hammer on an anvil",
    "a DSLR photo of an origami motorcycle",
    "a DSLR photo of an ornate silver gravy boat sitting on a patterned tablecloth",
    "a DSLR photo of an overstuffed pastrami sandwich",
    "a DSLR photo of an unstable rock cairn in the middle of a stream",
    "a DSLR photo of a pair of headphones sitting on a desk",
    "a DSLR photo of a pair of tan cowboy boots, studio lighting, product photography",
    "a DSLR photo of a peacock on a surfboard",
    "a DSLR photo of a pigeon reading a book",
    "a DSLR photo of a piglet sitting in a teacup",
    "a DSLR photo of a pig playing a drum set",
    "a DSLR photo of a pile of dice on a green tabletop next to some playing cards",
    "a DSLR photo of a pirate collie dog, high resolution",
    "a DSLR photo of a plate of fried chicken and waffles with maple syrup on them",
    "a DSLR photo of a plate piled high with chocolate chip cookies",
    "a DSLR photo of a plush t-rex dinosaur toy, studio lighting, high resolution",
    "a DSLR photo of a plush triceratops toy, studio lighting, high resolution",
    "a DSLR photo of a pomeranian dog",
    "a DSLR photo of a porcelain dragon",
    "a DSLR photo of a praying mantis wearing roller skates",
    "a DSLR photo of a puffin standing on a rock",
    "a DSLR photo of a pug made out of metal",
    "a DSLR photo of a pug wearing a bee costume",
    "a DSLR photo of a quill and ink sitting on a desk",
    "a DSLR photo of a raccoon stealing a pie",
    "a DSLR photo of a red cardinal bird singing",
    "a DSLR photo of a red convertible car with the top down",
    "a DSLR photo of a red-eyed tree frog",
    "a DSLR photo of a red pickup truck driving across a stream",
    "a DSLR photo of a red wheelbarrow with a shovel in it",
    "a DSLR photo of a roast turkey on a platter",
    "a DSLR photo of a robot and dinosaur playing chess, high resolution",
    "a DSLR photo of a robot arm picking up a colorful block from a table",
    "a DSLR photo of a robot cat knocking over a chess piece on a board",
    "a DSLR photo of a robot dinosaur",
    "a DSLR photo of a robot made out of vegetables",
    "a DSLR photo of a robot stegosaurus",
    "a DSLR photo of a robot tiger",
    "a DSLR photo of a rolling pin on top of bread dough",
    "a DSLR photo of a sheepdog running",
    "a DSLR photo of a shiba inu playing golf wearing tartan golf clothes and hat",
    "a DSLR photo of a shiny silver robot cat",
    "a DSLR photo of a silverback gorilla holding a golden trophy",
    "a DSLR photo of a silver humanoid robot flipping a coin",
    "a DSLR photo of a small cherry tomato plant in a pot with a few red tomatoes growing on it",
    "a DSLR photo of a small saguaro cactus planted in a clay pot",
    "a DSLR photo of a Space Shuttle",
    "a DSLR photo of a squirrel dressed like a clown",
    "a DSLR photo of a squirrel flying a biplane",
    "a DSLR photo of a squirrel giving a lecture writing on a chalkboard",
    "a DSLR photo of a squirrel holding a bowling ball",
    "a DSLR photo of a squirrel-lizard hybrid",
    "a DSLR photo of a squirrel made out of fruit",
    "a DSLR photo of a squirrel-octopus hybrid",
    "a DSLR photo of a stack of pancakes covered in maple syrup",
    "a DSLR photo of a steam engine train, high resolution",
    "a DSLR photo of a steaming basket full of dumplings",
    "a DSLR photo of a steaming hot plate piled high with spaghetti and meatballs",
    "a DSLR photo of a steampunk space ship designed in the 18th century",
    "a DSLR photo of a straw basket with a cobra coming out of it",
    "a DSLR photo of a swan and its cygnets swimming in a pond",
    "a DSLR photo of a tarantula, highly detailed",
    "a DSLR photo of a teal moped",
    "a DSLR photo of a teapot shaped like an elephant head where its snout acts as the spout",
    "a DSLR photo of a teddy bear taking a selfie",
    "a DSLR photo of a terracotta bunny",
    "a DSLR photo of a tiger dressed as a doctor",
    "a DSLR photo of a tiger made out of yarn",
    "a DSLR photo of a toilet made out of gold",
    "a DSLR photo of a toy robot",
    "a DSLR photo of a train engine made out of clay",
    "a DSLR photo of a tray of Sushi containing pugs",
    "a DSLR photo of a tree stump with an axe buried in it",
    "a DSLR photo of a turtle standing on its hind legs, wearing a top hat and holding a cane",
    "a DSLR photo of a very beautiful small organic sculpture made of fine clockwork and gears with tiny ruby bearings, very intricate, caved, curved. Studio lighting, High resolution, white background",
    "a DSLR photo of A very beautiful tiny human heart organic sculpture made of copper wire and threaded pipes, very intricate, curved, Studio lighting, high resolution",
    "a DSLR photo of a very cool and trendy pair of sneakers, studio lighting",
    "a DSLR photo of a vintage record player",
    "a DSLR photo of a wine bottle and full wine glass on a chessboard",
    "a DSLR photo of a wooden desk and chair from an elementary school",
    "a DSLR photo of a yorkie dog eating a donut",
    "a DSLR photo of a yorkie dog wearing extremely cool sneakers",
    "a DSLR photo of baby elephant jumping on a trampoline",
    "a DSLR photo of cat wearing virtual reality headset in renaissance oil painting high detail caravaggio",
    "a DSLR photo of edible typewriter made out of vegetables",
    "a DSLR photo of Mont Saint-Michel, France, aerial view",
    "a DSLR photo of Mount Fuji, aerial view",
    "a DSLR photo of Neuschwanstein Castle, aerial view",
    "A DSLR photo of   pyramid shaped burrito with a slice cut out of it",
    "a DSLR photo of the Imperial State Crown of England",
    "a DSLR photo of the leaning tower of Pisa, aerial view",
    "a DSLR photo of the Statue of Liberty, aerial view",
    "a DSLR photo of Two locomotives playing tug of war",
    "a DSLR photo of two macaw parrots sharing a milkshake with two straws",
    "a DSLR photo of Westminster Abbey, aerial view",
    "a ficus planted in a pot",
    "a flower made out of metal",
    "a fluffy cat lying on its back in a patch of sunlight",
    "a fox and a hare tangoing together",
    "a fox holding a videogame controller",
    "a fox playing the cello",
    "a frazer nash super sport car",
    "a freshly baked loaf of sourdough bread on a cutting board",
    "a goat drinking beer",
    "a golden goblet, low poly",
    "a green dragon breathing fire",
    "a green tractor farming corn fields",
    "a highland cow",
    "a hotdog in a tutu skirt",
    "a humanoid robot laying on the couch while on a laptop",
    "a humanoid robot playing the violin",
    "a humanoid robot sitting looking at a Go board with some pieces on it",
    "a human skeleton drinking a glass of red wine",
    "a human skull with a vine growing through one of the eye sockets",
    "a kitten looking at a goldfish in a bowl",
    "a lemur drinking boba",
    "a lemur taking notes in a journal",
    "a lionfish",
    "a llama wearing a suit",
    "a marble bust of a mouse",
    "a metal sculpture of a lion's head, highly detailed",
    "a mojito in a beach chair",
    "a monkey-rabbit hybrid",
    "an airplane made out of wood",
    "an amigurumi bulldozer",
    "An anthropomorphic tomato eating another tomato",
    "an astronaut playing the violin",
    "an astronaut riding a kangaroo",
    "an English castle, aerial view",
    "an erupting volcano, aerial view",
    "a nest with a few white eggs and one golden egg",
    "an exercise bike",
    "an iridescent metal scorpion",
    "An octopus and a giraffe having cheesecake",
    "an octopus playing the harp",
    "an old vintage car",
    "an opulent couch from the palace of Versailles",
    "an orange road bike",
    "an orangutan holding a paint palette in one hand and a paintbrush in the other",
    "an orangutan playing accordion with its hands spread wide",
    "an orangutan using chopsticks to eat ramen",
    "an orchid flower planted in a clay pot",
    "a palm tree, low poly 3d model",
    "a panda rowing a boat in a pond",
    "a panda wearing a necktie and sitting in an office chair",
    "A Panther De Ville car",
    "a pig wearing a backpack",
    "a plate of delicious tacos",
    "a plush dragon toy",
    "a plush toy of a corgi nurse",
    "a rabbit, animated movie character, high detail 3d model",
    "a rabbit cutting grass with a lawnmower",
    "a red eyed tree frog, low poly",
    "a red panda",
    "a ripe strawberry",
    "a roulette wheel",
    "a shiny red stand mixer",
    "a silver platter piled high with fruits",
    "a sliced loaf of fresh bread",
    "a snail on a leaf",
    "a spanish galleon sailing on the open sea",
    "a squirrel dressed like Henry VIII king of England",
    "a squirrel gesturing in front of an easel showing colorful pie charts",
    "a squirrel wearing a tuxedo and holding a conductor's baton",
    "a team of butterflies playing soccer on a field",
    "a teddy bear pushing a shopping cart full of fruits and vegetables",
    "a tiger dressed as a military general",
    "a tiger karate master",
    "a tiger playing the violin",
    "a tiger waiter at a fancy restaurant",
    "a tiger wearing a tuxedo",
    "a t-rex roaring up into the air",
    "a turtle standing on its hind legs, wearing a top hat and holding a cane",
    "a typewriter",
    "a walrus smoking a pipe",
    "a wedge of cheese on a silver platter",
    "a wide angle DSLR photo of a colorful rooster",
    "a wide angle DSLR photo of a humanoid banana sitting at a desk doing homework",
    "a wide angle DSLR photo of a mythical troll stirring a cauldron",
    "a wide angle DSLR photo of a squirrel in samurai armor wielding a katana",
    "a wide angle zoomed out DSLR photo of A red dragon dressed in a tuxedo and playing chess. The chess pieces are fashioned after robots",
    "a wide angle zoomed out DSLR photo of a skiing penguin wearing a puffy jacket",
    "a wide angle zoomed out DSLR photo of zoomed out view of Tower Bridge made out of gingerbread and candy",
    "a woolly mammoth standing on ice",
    "a yellow schoolbus",
    "a zoomed out DSLR photo of a 3d model of an adorable cottage with a thatched roof",
    "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
    "a zoomed out DSLR photo of a baby dragon",
    "a zoomed out DSLR photo of a baby monkey riding on a pig",
    "a zoomed out DSLR photo of a badger wearing a party hat and blowing out birthday candles on a cake",
    "a zoomed out DSLR photo of a beagle eating a donut",
    "a zoomed out DSLR photo of a bear playing electric bass",
    "a zoomed out DSLR photo of a beautifully carved wooden knight chess piece",
    "a zoomed out DSLR photo of a beautiful suit made out of moss, on a mannequin. Studio lighting, high quality, high resolution",
    "a zoomed out DSLR photo of a blue lobster",
    "a zoomed out DSLR photo of a blue tulip",
    "a zoomed out DSLR photo of a bowl of cereal and milk with a spoon in it",
    "a zoomed out DSLR photo of a brain in a jar",
    "a zoomed out DSLR photo of a bulldozer made out of toy bricks",
    "a zoomed out DSLR photo of a cake in the shape of a train",
    "a zoomed out DSLR photo of a chihuahua lying in a pool ring",
    "a zoomed out DSLR photo of a chimpanzee dressed as a football player",
    "a zoomed out DSLR photo of a chimpanzee holding a cup of hot coffee",
    "a zoomed out DSLR photo of a chimpanzee wearing headphones",
    "a zoomed out DSLR photo of a colorful camping tent in a patch of grass",
    "a zoomed out DSLR photo of a complex movement from an expensive watch with many shiny gears, sitting on a table",
    "a zoomed out DSLR photo of a construction excavator",
    "a zoomed out DSLR photo of a corgi wearing a top hat",
    "a zoomed out DSLR photo of a corn cob and a banana playing poker",
    "a zoomed out DSLR photo of a dachsund riding a unicycle",
    "a zoomed out DSLR photo of a dachsund wearing a boater hat",
    "a zoomed out DSLR photo of a few pool balls sitting on a pool table",
    "a zoomed out DSLR photo of a fox working on a jigsaw puzzle",
    "a zoomed out DSLR photo of a fresh cinnamon roll covered in glaze",
    "a zoomed out DSLR photo of a green tractor",
    "a zoomed out DSLR photo of a greyhound dog racing down the track",
    "a zoomed out DSLR photo of a group of squirrels rowing crew",
    "a zoomed out DSLR photo of a gummy bear driving a convertible",
    "a zoomed out DSLR photo of a hermit crab with a colorful shell",
    "a zoomed out DSLR photo of a hippo biting through a watermelon",
    "a zoomed out DSLR photo of a hippo made out of chocolate",
    "a zoomed out DSLR photo of a humanoid robot lying on a couch using a laptop",
    "a zoomed out DSLR photo of a humanoid robot sitting on a chair drinking a cup of coffee",
    "a zoomed out DSLR photo of a human skeleton relaxing in a lounge chair",
    "a zoomed out DSLR photo of a kangaroo sitting on a bench playing the accordion",
    "a zoomed out DSLR photo of a kingfisher bird",
    "a zoomed out DSLR photo of a ladybug",
    "a zoomed out DSLR photo of a lion's mane jellyfish",
    "a zoomed out DSLR photo of a lobster playing the saxophone",
    "a zoomed out DSLR photo of a majestic sailboat",
    "a zoomed out DSLR photo of a marble bust of a cat, a real mouse is sitting on its head",
    "a zoomed out DSLR photo of a marble bust of a fox head",
    "a zoomed out DSLR photo of a model of a house in Tudor style",
    "a zoomed out DSLR photo of a monkey-rabbit hybrid",
    "a zoomed out DSLR photo of a monkey riding a bike",
    "a zoomed out DSLR photo of a mountain goat standing on a boulder",
    "a zoomed out DSLR photo of a mouse holding a candlestick",
    "a zoomed out DSLR photo of an adorable kitten lying next to a flower",
    "a zoomed out DSLR photo of an all-utility vehicle driving across a stream",
    "a zoomed out DSLR photo of an amigurumi motorcycle",
    "a zoomed out DSLR photo of an astronaut chopping vegetables in a sunlit kitchen",
    "a zoomed out DSLR photo of an egg cracked open with a newborn chick hatching out of it",
    "a zoomed out DSLR photo of an expensive office chair",
    "a zoomed out DSLR photo of an origami bulldozer sitting on the ground",
    "a zoomed out DSLR photo of an origami crane",
    "a zoomed out DSLR photo of an origami hippo in a river",
    "a zoomed out DSLR photo of an otter lying on its back in the water holding a flower",
    "a zoomed out DSLR photo of a pair of floating chopsticks picking up noodles out of a bowl of ramen",
    "a zoomed out DSLR photo of a panda throwing wads of cash into the air",
    "a zoomed out DSLR photo of a panda wearing a chef's hat and kneading bread dough on a countertop",
    "a zoomed out DSLR photo of a pigeon standing on a manhole cover",
    "a zoomed out DSLR photo of a pig playing the saxophone",
    "a zoomed out DSLR photo of a pile of dice on a green tabletop",
    "a zoomed out DSLR photo of a pita bread full of hummus and falafel and vegetables",
    "a zoomed out DSLR photo of a pug made out of modeling clay",
    "a zoomed out DSLR photo of A punk rock squirrel in a studded leather jacket shouting into a microphone while standing on a stump and holding a beer",
    "a zoomed out DSLR photo of a rabbit cutting grass with a lawnmower",
    "a zoomed out DSLR photo of a rabbit digging a hole with a shovel",
    "a zoomed out DSLR photo of a raccoon astronaut holding his helmet",
    "a zoomed out DSLR photo of a rainforest bird mating ritual dance",
    "a zoomed out DSLR photo of a recliner chair",
    "a zoomed out DSLR photo of a red rotary telephone",
    "a zoomed out DSLR photo of a robot couple fine dining",
    "a zoomed out DSLR photo of a rotary telephone carved out of wood",
    "a zoomed out DSLR photo of a shiny beetle",
    "a zoomed out DSLR photo of a silver candelabra sitting on a red velvet tablecloth, only one candle is lit",
    "a zoomed out DSLR photo of a squirrel DJing",
    "a zoomed out DSLR photo of a squirrel dressed up like a Victorian woman",
    "a zoomed out DSLR photo of a table with dim sum on it",
    "a zoomed out DSLR photo of a tiger dressed as a maid",
    "a zoomed out DSLR photo of a tiger dressed as a military general",
    "a zoomed out DSLR photo of a tiger eating an ice cream cone",
    "a zoomed out DSLR photo of a tiger wearing sunglasses and a leather jacket, riding a motorcycle",
    "a zoomed out DSLR photo of a toad catching a fly with its tongue",
    "a zoomed out DSLR photo of a wizard raccoon casting a spell",
    "a zoomed out DSLR photo of a yorkie dog dressed as a maid",
    "a zoomed out DSLR photo of cats wearing eyeglasses",
    "a zoomed out DSLR photo of miniature schnauzer wooden sculpture, high quality studio photo",
    "A zoomed out DSLR photo of   phoenix made of splashing water ",
    "a zoomed out DSLR photo of Sydney opera house, aerial view",
    "a zoomed out DSLR photo of two foxes tango dancing",
    "a zoomed out DSLR photo of two raccoons playing poker",
    "Chichen Itza, aerial view",
    "  Coffee cup with many holes",
    "fries and a hamburger",
    "  Luminescent wild horses",
    "Michelangelo style statue of an astronaut",
    "Michelangelo style statue of dog reading news on a cellphone",
    "the titanic, aerial view",
    "two gummy bears playing dominoes",
    "two macaw parrots playing chess",
    "Wedding dress made of tentacles"
]

gpteval3d_prompts = [
    "A torn hat",
    "A soft sofa",
    "A rusty boat",
    "A rough rock",
    "A thorny rose",
    "A brick house",
    "A crying sofa",
    "A sleeping cat",
    "A twisted tower",
    "A shouting leaf",
    "A rubbery cactus",
    "A wooden bicycle",
    "A dancing elephant",
    "A pen leaking blue ink",
    "A pair of worn-out shoes",
    "A lamp casting a warm glow",
    "A book with a leather cover",
    "A teddy bear with a red bow",
    "A book left on a park bench",
    "Four ripe apples in a basket",
    "A boat floating on calm water",
    "A chair made from polished oak",
    "A bicycle leaning against a wall",
    "An origami crane made from a map",
    "A mug filled with steaming coffee",
    "A book bound in mysterious symbols",
    "A dog creating sand art on a beach",
    "Three vibrant balloons tied together",
    "A cat with two different colored eyes",
    "An orange tabby cat shaped cookie jar",
    "A quartet of mugs that sing in harmony",
    "Floating bonsai tree, roots in mid-air",
    "Brown rabbit nibbling on a clover leaf",
    "An embroidered silk pillow with tassels",
    "A guitar resting against an old oak tree",
    "A pen sitting atop a pile of manuscripts",
    "A bicycle that leaves a trail of flowers",
    "A weathered hiking backpack with patches",
    "Sand hourglass, sand glitters like stars",
    "A velvet diary, locks with a fingerprint",
    "Spotted ladybug crawling on a green leaf",
    "Gray squirrel with an acorn in its mouth",
    "A teddy bear, fur matted, one eye missing",
    "A golden retriever plush toy, floppy-eare",
    "A carved wooden bear with a salmon in mouth",
    "A cat pondering the mysteries of the universe",
    "A lamp casting shadows on an old, forgotten map",
    "Orange monarch butterfly resting on a dandelion",
    "Clownfish peeking out from sea anemone tendrils",
    "Caterpillar with a keyboard pattern on its back",
    "A quill pen, feather shifts through rainbow hues",
    "Green tree frog clinging to a rain-soaked window",
    "An old-fashioned rotary phone with a tangled cord",
    "Tortoise with a shell that looks like stained glass",
    "A plush octopus whose arms are gently waving pencils",
    "An assortment of vintage, fragrant perfumes on display",
    "A group of vibrant, chattering parrots perched together",
    "Swan with feathers resembling soft, white origami folds",
    "A quaint, little house nestled at the end of a winding path",
    "Flamingo balancing on a sphere instead of standing in water",
    "A bright yellow rubber duck gently floats in a sudsy bathtub",
    "A collection of fresh vegetables arranged in a wicker basket",
    "An ancient, weathered statue, now covered in a blanket of moss",
    "A mesmerizing dance performed by a kaleidoscope of butterflies",
    "A lone, ancient tree stands tall in the middle of a quiet field",
    "A plush teddy bear, sitting alone with a slight tear in its seam",
    "A pair of hiking boots caked with mud at the doorstep of a cabin",
    "A worn leather recliner with a knitted throw draped over the back",
    "A small, intricately carved antique wooden box filled with mystery",
    "Frog with a translucent skin displaying a mechanical heart beating",
    "A small, rustic cabin sits alone in a peaceful, snow-covered forest",
    "Jellyfish with bioluminescent tentacles shaped like lightning bolts",
    "A floating teapot, pouring a stream of endless, steaming jasmine tea",
    "A smoldering campfire under a clear starry night, embers glowing softly",
    "A smartphone with a cracked screen lying on a coffee-stained office desk",
    "An ice cream scoop that serves up scoops of cloud fluff instead of ice cream",
    "A velvet-lined violin case, which opens to reveal a garden of miniature roses",
    "A hammock strung between two skyscrapers, swaying high above a neon cityscape",
    "A delicate porcelain teacup, painted with intricate flowers, rests on a saucer",
    "A sleek red sports car with chrome finishes parked by a bustling city sidewalk",
    "An ensemble of jellyfish-like hanging lamps, pulsing with soft bioluminescence",
    "A cluster of tents pitched near a forest, campfire smoke curling into the evening sky",
    "A colorful kite tangled in the branches of an oak tree, fabric fluttering in the wind",
    "A dragon-shaped kite, with scales that shimmer in the sunlight as it dances in the wind",
    "A stone bridge arching over a babbling brook, encrusted with moss and echoing with stories",
    "A sequence of street lamps, casting pools of light on cobblestone paths as twilight descends",
    "A bouquet of metallic flowers that bloom in response to moonlight, shimmering silver at night",
    "An array of small, solid, symmetrical, pastel-colored eggs, each revealing a miniature, enchanted forest scene when cracked open",
    "A heavy, layered, asymmetrical winter quilt, with a patchwork of plaid fabrics in reds and greens, folded at the foot of a well-made bed",
    "Various hollow, asymmetrical, textured seashells, collected in a sand-filled, clear glass jar with a twine-tied neck, displayed on a windowsill",
    "An old, solid, asymmetrical bronze bell, its contours irregular from centuries of use, with a green patina, sitting silent in an abandoned temple",
    "A small, hollow, asymmetrical birdhouse, painted in cheerful colors, with a round entrance and a tiny perch, swaying gently in a backyard apple tree",
    "An ensemble of hollow, irregularly shaped musical instruments, including a saxophone, a violin, and a drum, resting on a stage before a jazz concert",
    "An assortment of solid, symmetrical, smooth marbles, each one a different color with a unique swirl pattern, scattered playfully across a hardwood floor",
    "A medium-sized, layered, radially symmetrical conch shell, with a rough texture on the outside, fading from pink to cream, sitting alone on a sandy beach",
    "Several large, solid, symmetrical hay bales, with a rough, golden texture, scattered across a rural, open field, with the setting sun casting long shadows",
    "An old, layered, asymmetrical lantern, with a patina copper finish and translucent panes that flicker with bioluminescent light from cultured algae within",
    "A collection of solid, irregularly shaped hand tools, with wooden handles and metal ends, well-used and slightly rusty, hanging on a pegboard in a workshop",
    "Several solid, spherical, weathered cannonballs, with a rough cast-iron texture, lying beside a rusted cannon in a historical fort overlooking a serene bay",
    "A small, solid, geometrically spherical, metallic orb, with a glossy ruby finish, nestled in a nest of black velvet, untouched and gleaming under a spotlight",
    "An intricately carved, solid, wooden figurine, with jagged contours depicting an ancient deity, the wood grain visible under a matte finish, on a stone altar",
    "A large, multi-layered, symmetrical wedding cake, with smooth fondant, delicate piping, and lifelike sugar flowers in full bloom, displayed on a silver stand",
    "A large, hollow, asymmetrically shaped amphitheater, with jagged stone seating, nestled in a natural landscape, a classical play being performed as the sun sets",
    "A medium-sized, hollow, asymmetrical teapot, crafted to look like a slumbering dragon, with a scaly, rough texture and smoke gently wafting from its snout-spout",
    "A solid, smooth, symmetrical porcelain teapot, with a cobalt blue dragon design, steam rising from the spout, suggesting it's just been filled with boiling water",
    "A compact, cylindrical, vintage pepper mill, with a polished, ornate brass body, slightly worn from use, placed beside a porcelain plate on a checkered tablecloth",
    "Several large, solid, cube-shaped parcels, wrapped in brown paper and tied with string, each labeled with a different destination, awaiting dispatch in a post office",
    "An oversized, porous, sphere-shaped birdcage, made of woven golden wires, with a matte finish, housing a small, mechanical, singing bird that flutters in a lifelike manner",
    "A small, solid, radially symmetrical, iridescent abalone shell, with jagged contours, hosting a miniature, tranquil Zen garden complete with tiny, raked sand and micro bonsai",
    "A solid, symmetrical, smooth stone fountain, with water cascading over its edges into a clear, circular pond surrounded by blooming lilies, in the center of a sunlit courtyard"
]

ds = load_dataset("rubend18/DALL-E-Prompts-OpenAI-ChatGPT")

length = 15
filtered_captions_DALLE = []
for item in tqdm(ds['train']):
    prompt = item['Prompt']
    if len(prompt.split(' ')) < length:
        filtered_captions_DALLE.append(prompt)
filtered_captions_DALLE = list(set(filtered_captions_DALLE))

filtered_captions_midjourney_v1 = []
ds = load_dataset("CortexLM/midjourney-v6")
for caption in tqdm(ds['train']['prompt']):
    if len(caption.split(' ')) < length:
        filtered_captions_midjourney_v1.append(caption)
filtered_captions_midjourney_v1 = list(set(filtered_captions_midjourney_v1))


ds = load_dataset("wanng/midjourney-v5-202304-clean")
filtered_captions_midjourney_v2 = []
for caption in tqdm(ds['train']['clean_prompts']):
    if len(caption.split(' ')) < length:
        filtered_captions_midjourney_v2.append(caption)
filtered_captions_midjourney_v2  = list(set(filtered_captions_midjourney_v2))

ds = load_dataset("brivangl/midjourney-v6-llava",columns=["prompt"],streaming=True)
filtered_captions_midjourney_v3 = []
for caption in tqdm(ds['train']):
    if len(caption['prompt'].split(' ')) < length:
        filtered_captions_midjourney_v3.append(caption['prompt'])
filtered_captions_midjourney_v3 = list(set(filtered_captions_midjourney_v3))


# save to file
save_dict = {
    "train": [],
    "val": [],
    "test": []
}

save_dict["train"] = filtered_captions_DALLE[50:] + filtered_captions_midjourney_v1[50:] + filtered_captions_midjourney_v2[50:] + filtered_captions_midjourney_v3[50:]
save_dict["train"] = list(set(save_dict["train"]))
save_dict["val"] = dreamfusion_prompts[:50] + gpteval3d_prompts[:50] + filtered_captions_DALLE[:50] + filtered_captions_midjourney_v1[:50] + filtered_captions_midjourney_v2[:50] + filtered_captions_midjourney_v3[:50]
save_dict["test"] = dreamfusion_prompts + gpteval3d_prompts

save_path = "load/DALLE_Midjourney_{}_prompt_library.json".format(len(save_dict["train"]))
with open(save_path, 'w') as f:
    json.dump(save_dict, f, indent=4)
    