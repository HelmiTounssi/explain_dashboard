# explain-dashboard : Home Credit Prediction

## test en local avec docker
docker build -t explain-dashboard .
docker run -p 9050:9050 explain-dashboard:latest

## github actions  fait le ci/cd en google cloud

 1-creer un cluster kubernetes autopilot-cluster-1  
 2- creer Artifact registery  home-credit-repo
 3- creer cloud storage Buckets  : data-model-home-credit 
 4- faire un push et github action build.xml creer le livrable docker ,push dans le registry puis instance avec ressources.yaml dans kubernetes .
 5-ajouter le token google .json et le id de projet  dans secert action dans github.
