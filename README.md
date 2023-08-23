# forest_guardian

## Inspiration
The Amazon forest is a critical ecological powerhouse, stabilizing the climate, harboring immense biodiversity, and supporting local communities. Preventing deforestation and increasing reforestation are imperative to preserve these essential ecosystem services, combat climate change, and protect the intricate web of life that depends on this unique biome. Thus this inspired us to make the Amazon Guardian Project to protect the Amazon forest using the AI technology.

## What it does
This is a comprehensive AI-driven project ensuring forest protection through CV, ML, and LLM techniques. Detecting deforestation, forest- fires, and illegal activities, fostering eco-awareness and ethical use of forest resources.

### The functions of this project:
- detect the signs of degradation of the forest 
- Make attempts to prevent the prevent the hazards
- Encourage sustainable approach to protect the Amazon and try to make people aware of the consequences of amazon degradation

### What are the features of this project
-  Green Check
Tracking Forest Cover Changes via Advanced Computer Vision(Image segmentation). AI tracks forest changes via images from satellites/drones. Detects cover increase/decrease, aids conservation efforts. The correct aerial image of the forest cover is uploaded and the model compares with the previous images to detect the decrease. 
The ML model built using ##  [ Brazilian Amazon Rainforest Degradation 1999-2019](https://www.kaggle.com/datasets/mbogernetto/brazilian-amazon-rainforest-degradation) dataset, predicts the degradation in the different places. This data along with the dressed area value will be passed to the LLM which will tell the consequences of the actions and make the people aware to take immediate actions.
- Amazon FireWatch
Advanced Fire Detection System using Computer Vision to aid early detection of the forest fires.
Here the ML model built on  [ Brazilian Amazon Rainforest Degradation 1999-2019](https://www.kaggle.com/datasets/mbogernetto/brazilian-amazon-rainforest-degradation) dataset for fire prediction to make predict firespots in Amazon area. and show it to the authorities to take the necessary actions to prevent forest fires.
![fire](https://github.com/lakshmishreea122003/aaaaaa/blob/main/fire.jpg)
- Evacuation
Use AI and ML to integrate fire prediction and deforestation data to recommend vital evacuations, ensuring safety from both wildfires and human impact. This is feature is inspired by the fact that the main cause for degradation is hunams in that region who contribute to deforestation by agriculture land expansion, forest fires due to slash and burn methods and more reasons
At times evacuation of people from a certain region is crucial to prevent further degration in the sensitive regions. But not easy. It takes time for authorities to execute evacuation plane. Thus the predictive ML models can be useful.
Here ML model  to predict firespots in Amazon area and ML model to predict degradation in 
Amazon area, both built on [ Brazilian Amazon Rainforest Degradation 1999-2019](https://www.kaggle.com/datasets/mbogernetto/brazilian-amazon-rainforest-degradation) will make the desired prediction. This data will be passed to the LLM which will suggest which place to be evacuated and why.
![evacuation](https://github.com/lakshmishreea122003/aaaaaa/blob/main/Evacuation.jpg)
- CattleCount Alert
Use Computer Vision for Automated Monitoring for Herd Size Management to prevent deforestation due to grazing. Here the model receives the satellite/drone images, checks if the cattel count increases the maximum number that was gives as input. If exceeds then authorities get an alert message. This way deforestation due to cattel grazing can be prevented in the sensitive areas.
- Vehicle Detection & Access
This project ensures secure access of vehicles in forests by identifying vehicles and validating license plates, preventing unauthorized entry to sensitive forest areas, and deterring illegal deforestation activities effectively. Logging practices can be discouraged via this feature by not allowing unauthorised vehicles in the sensitive regions of forest. Thus targeting illegal transport of resources problem.
- Sustainable Scope
Amazon Forest Sustainability Analysis & Alternatives for Business and Agriculture using LLM. A sustainable approach is required to protect the amazon forest. BY immediately all the resource usage cannot come to a stop. So we can use LLM to check whether the business plan or the agriculture decisions of expansion or any related plan is sustainable or no. The user can either give text input of the plan or a pdf about the plan report. The LLM model will analysise and let us know about the sustainability of the plan.

## How we built it
Streamlit = to built the web app
Langchain = use langchain to integrate the project with OpenAI API and harness the power of LLM models in our project
Computer Vision  = Used to detect forest first, image segmentation for forest cover decrease detection, cattle count, vehicle access detection 
Machine learning = to make predictive models
LLM = Consider the predictions and give required suggestions and also to get data related to the consequences.

## Challenges we ran into
Knowing the causes for the degration of the amazon forest and finding ways to tackle them using AI technology was challenging for me.

## Accomplishments that we're proud of
We were finally able to make a project to protect the Amazon forest. We have taken a holistic approach to use AI and make attempts to protect the Amazon forest

## What we learned
We learnt to use AI in the right manner to make this project to contribute to humanity's battle to combat climate change.

## What's next for Amazon Guardian
##############
