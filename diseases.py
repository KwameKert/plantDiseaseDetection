
def load_diseases():
    lists = ['Apple___Apple scab', 'Apple___Black rot', 'Apple___Cedar apple rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry(including sour)___Powdery mildew', 'Cherry(including sour)___healthy', 'Corn (maize)___Cercospora leaf spot Gray leaf spot', 'Corn (maize)___Common rust ', 'Corn (maize)___Northern Leaf Blight', 'Corn (maize)___healthy', 'Grape___Black rot', 'Grape___Esca (Black Measles)', 'Grape___Leaf blight (Isariopsis Leaf Spot)', 'Grape___healthy', 'Orange___Haunglongbing (Citrus greening)', 'Peach___Bacterial spot', 'Peach___healthy', 'Pepper bell___Bacterial spot', 'Pepper ___healthy', 'Potato___Early blight', 'Potato___Late blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early blight', 'Tomato___Late blight', 'Tomato___Leaf Mold', 'Tomato___Septoria leaf spot', 'Tomato___Spider mites Two spotted spider mite', 'Tomato___Target Spot', 'Tomato___Tomato Yellow Leaf Curl Virus', 'Tomato___Tomato mosaic virus', 'Tomato___healthy']

    return lists



def load_remedies():

     remedies = ['Apply as a soil drench or foliar spray (3-4 tsp/ gallon of water) to prevent and attack fungal problems.\n For best results, apply as a protective spray (2.5 oz/ gallon) early in the season.', 'Remove the cankers by pruning at least 15 inches below the end and burn or bury them.\n Also take preventative care with new season pruning and burn them, too \n Trim your trees when they are dormant, disinfect tools between cuts, and burn the branches and leaves', 'If you see the lesions on the apple leaves or fruit, it is too late to control the fungus. In that case, you should focus on purging infected leaves and fruit from around your tree. \nSpraying apple trees with copper can be done to treat cedar apple rust and prevent other fungal infections','healthy','healthy','Spray on plants every one to two weeks. Potassium bicarbonate– Similar to baking soda, this has the unique advantage of actually eliminating powdery mildew once it\'s there. Potassium bicarbonate is a contact fungicide which kills the powdery mildew spores quickly. In addition, it\'s approved for use in organic growing.','healthy','During the growing season, foliar fungicides can be used to manage gray leaf spot outbreaks \n Farmers must consider the cost of the application and market value of their corn before determining if fungicides will be an economical solution to GLS. ', 'To reduce the incidence of corn rust, plant only corn that has resistance to the fungus. Resistance is either in the form of race-specific resistance or partial rust resistance. In either case, no sweet corn is completely resistant. If the corn begins to show symptoms of infection, immediately spray with a fungicide. The fungicide is most effective when started at the first sign of infection. Two applications may be necessary. Contact your local extension office for advice regarding specific fungicides and their uses. \n Read more at Gardening Know How: Sweet Corn Rust Treatment – Learn About Corn Rust Fungus Control https://www.gardeningknowhow.com/edible/vegetables/corn/corn-rust-fungus-control.htm', 'The fungus overwinters in plant material, so it is also important to manage infected plants. Tilling the corn into the soil is one strategy, but with a small garden it may make more sense to just remove and destroy the affected plants. Treating northern corn leaf blight involves using fungicides', 'healthy', 'The best time to treat black rot of grapes is between bud break until about four weeks after bloom; treating outside of this window is likely to end in frustration. However, if you want to try, captan and myclobutanil are the fungicides of choice \n Black rot spores love moisture, so you want to reduce the amount of moisture held in the canopy by providing great air circulation. During dormancy, prune heavily keeping only a few healthy, strong canes from last year\'s growth. Prune out any diseased parts of the vine too.', 'No remedy', 'No remedy', 'healthy', 'Bactericides are a topical treatment aimed at slowing the bacteria that cause citrus greening. \n It is incredibly important to remove trees that have citrus greening disease.', 'Where disease incidence is high, fungicides may be applied. On peach trees, a dormant spray of copper fungicide in late fall will work well. \n Keep the ground free of leaves and debris, especially over the winter. Prune and destroy infected plant parts as soon as you see them.', 'healthy', 'Soil should be well drained, but be sure to maintain adequate moisture either with mulch or plastic covering. \n Water one to two inches per week, but remember that peppers are extremely heat sensitive. If you live in a warm or desert climate, watering everyday may be necessary. Fertilize after the first fruit set.','healthy', 'Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing is more resistant than early maturing varieties. \n Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible.', 'As soon as potato tops stop growing and lower leaves turn yellow, protecting tubers against late blight is important. If there is visible late blight infestation it is recommended to apply fungicides with a spore-killing effect (fluazinam-containing fungicides, Ranman Top) mainly.', 'healthy','healthy','healthy','The first step in treating squash blossom blight is to remove all infected fruits and vines to prevent the spread of the disease. Fungicides, such as liquid copper, are the most common treatment for squash blossom blight. To use this type of treatment, apply the liquid solution to the leaves and vines of the affected plant using a concentration of 1/2 to 2 fluid ounces of the solution per 1 gallon of water. \n Apply the fungicide treatment at the first sign of blight and repeat the application seven to 10 days later, following a rain. To use fungicides as a preventative, apply the solution before a long period of wet weather is predicted.  ', 'Remove foliage and crop residues after picking or at renovation to remove inoculum and delay disease increase in late summer and fall. Fungicide treatments are effective during the flowering period, and during late summer and fall.', 'healthy', '    a. A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit. \n Since water movement spreads the bacteria from diseased to healthy plants, workers and farm equipment should be kept out of fields when fields are wet, because the disease will spread readily under wet conditions. The traditional recommendation for bacterial spot control consists of copper and maneb or mancozeb.', 'Use pathogen-free seed, or collect seed only from disease-free plants. Rotate out of tomatoes and related crops for at least two years. Control susceptible weeds such as black nightshade and hairy nightshade, and volunteer tomato plants throughout the rotation. Fertilize properly to maintain vigorous plant growth.', 'Plant resistant cultivars when available. \n Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation. \n Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation. \n Destroy all tomato and potato debris after harvest (see Fall Garden Cleanup).', 'Applying fungicides when symptoms first appear can reduce the spread of the leaf mold fungus significantly. Several fungicides are labeled for leaf mold control on tomatoes and can provide good disease control if applied to all the foliage of the plant, especially the lower surfaces of the leaves.', 'Removing infected leaves. Remove infected leaves immediately, and be sure to wash your hands thoroughly before working with uninfected plants. \n Consider organic fungicide options. \n Consider chemical fungicides.', '    a. Prune leaves, stems and other infested parts of plants well past any webbing and discard in trash (and not in compost piles). Don’t be hesitant to pull entire plants to prevent the mites spreading to its neighbors. \n Use the Bug Blaster to wash plants with a strong stream of water and reduce pest numbers. \n Commercially available beneficial insects, such as ladybugs, lacewing and predatory mites are important natural enemies. For best results, make releases when pest levels are low to medium.', 'Do not plant new crops next to older ones that have the disease. \n Plant as far as possible from papaya, especially if leaves have small angular spots. \n Check all seedlings in the nursery, and throw away any with leaf spots.', 'Minimize Irrigation. Tomato plants have surprisingly low water needs and overwatering can promote disease. \n Water at Ground Level. \n Water in the Morning. \n Mulch. \n Remove Infected Leaves Immediately. \n Prune Out Dense Foliage. \n Keep Adjacent Vegetation Down. \n Disinfect Tomato Tools', 'There are no cures for viral diseases such as mosaic once a plant is infected. As a result, every effort should be made to prevent the disease from entering your garden. Fungicides will NOT treat this viral disease. Plant resistant varieties when available or purchase transplants from a reputable source.s', 'healthy']
    
     return remedies
