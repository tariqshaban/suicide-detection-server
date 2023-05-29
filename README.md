Embedding a Suicide Text Classification Model into Restful APIs
==============================
This is a supplementary submission of **final paper** for the **CIS737** course.

It contains the code necessary to host a local restful API that utilizes a transformer model to predict incoming request
values.

The model has the following hyperparameters:

* Sampling fraction, set to 1
* Validation ratio, set to 0.01
* Test ratio, set to 0.005
* Batch size, set to 2 (Due to computational limitation)
* Number of epochs, set to 300
* Early stopping value, set to 30
* Early stopping delta value, set to 0.001

The returned values are:

* The predicted label (suicide/non-suicide)
* The confidence in the prediction

Getting Started
------------
Clone the project from GitHub

`$ git clone https://github.com/tariqshaban/suicide-detection-server.git`

It is encouraged to refer to [Flask](https://flask.palletsprojects.com/) documentation.

You may need to configure the Python interpreter (depending on the used IDE).

You may encounter problems concerning CORS policy when the client and server are on the same machine.

No further configuration is required.

> **Note**: The model architecture located at `assets/model/tf_model.h5` is stored as a GitHub LFS; you may need to
> download the file separately.

--------