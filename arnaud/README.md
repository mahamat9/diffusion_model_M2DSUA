## Tentative de mise en oeuvre de la méthode de Sliced Score Matching (SSM)

Dans ce fichier notebook, nous avons essayé d'entrainer et de faire fonctionner un réseau de neuronnes de score via la méthode de Sliced Score Matching (SSM).
Nous avons utilisé PyTorch et le GPU T4 de google colab pour le faire. Le modèle de diffusion utilisé correspond à celui d'un processus d'Ornstein-Uhlenbeck, avec un temps discrétisé en n_steps étapes (configurable dans les paramètres du modèle de diffusion).  
Nous n'avons malheureusement pas réussi à générer des images avec, pour deux raisons majeurs : 
  - Lors du débruitage d'une image générée aléatoirement, nous avons systématiquement observé une explosion des valeurs vers l'infini, en ayant testé plusieurs valeurs de beta et sigma.
  - Comme énoncé dans le rapport et la présentation orale, le calcul de dérivée s'est avéré très coûteux. Nous avons essayé plusieurs méthodes qui ont été laissées en commentaires dans le code. Certaines ont été abandonnées car elles donnaient des pertes anormalement élevées par rapport aux autres (>1000 alors que ça ne dépasse pas 100 pour les autres). L'entrainement sur une époque pouvait prendre jusqu'à 5 minutes sur seulement 10 000 images MNIST.

Vous trouverez :
 - Au début du notebook les paramètres de reproductibilité et du modèle de diffusion. 
 - Et juste avant le bloc d'exécution les paramètres d'entrainement.