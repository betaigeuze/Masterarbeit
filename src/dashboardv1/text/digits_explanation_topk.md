<p class="text-font">
Above you can see the 10 most important features for the Random Forest. Compared to the Iris dataset, the values are much lower. This is because there are a lot more features in the digits dataset and each feature contributes a small amount to the decisions of the Random Forest. However, this is not the whole story, since there are 54 more features.
<br>
If you inspect the graph below, we learn, that there seem to be areas in the image that are completely irrelevant for the classification (at least for our Random Forest)! If we wanted to optimize our Random Forest, we could consider excluding these features in a future iteration of the model.
But let's dive a little deeper into the forest...
<br>