import matplotlib.pyplot as plt


label_names = [
        'chaotic',
        'fault',
        'other',
        'salt']

#label_names = [
#    'airplane',
#    'automobile',
#    'bird',
#    'cat',
#    'deer',
#    'dog',
#    'frog',
#    'horse',
#    'ship',
#    'truck'
#]

def plot_images(images, cls_true, cls_pred=None):

    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i, :, :, :], interpolation='spline16', cmap='gray')
        # get its equivalent class name
        cls_true_name = label_names[cls_true[i]]
            
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
