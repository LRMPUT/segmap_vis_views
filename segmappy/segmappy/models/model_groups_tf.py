import tensorflow as tf


# define the cnn model
def init_model(input_shape, input_shape_vis, n_classes, vis_views=False, triplet=0):
    with tf.name_scope("InputScope") as scope:
        cnn_input = tf.placeholder(
            dtype=tf.float32, shape=(None,) + input_shape + (1,), name="input"
        )

        cnn_input_vis = tf.placeholder(
            dtype=tf.float32, shape=(None,) + input_shape_vis + (3,), name="input_vis"
        )

    # base convolutional layers
    y_true = tf.placeholder(dtype=tf.float32, shape=(None, n_classes), name="y_true")

    scales = tf.placeholder(dtype=tf.float32, shape=(None, 3), name="scales")

    training = tf.placeholder_with_default(
        tf.constant(False, dtype=tf.bool), shape=(), name="training"
    )

    conv1 = tf.layers.conv3d(
        inputs=cnn_input,
        filters=32,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name="conv1",
    )

    pool1 = tf.layers.max_pooling3d(
        inputs=conv1, pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool1"
    )

    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name="conv3",
    )

    pool2 = tf.layers.max_pooling3d(
        inputs=conv2, pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2"
    )

    conv3 = tf.layers.conv3d(
        inputs=pool2,
        filters=64,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name="conv5",
    )

    conv3_out = tf.identity(conv3, name='conv3_out')

    if vis_views:
        # visual view convolution
        conv1_vis = tf.layers.conv2d(
                inputs=cnn_input_vis,
                filters=16,
                kernel_size=(5, 15),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv1_vis",
        )

        pool1_vis = tf.layers.max_pooling2d(
            inputs=conv1_vis, pool_size=(2, 2), strides=(2, 2), name="pool1_vis"
        )

        conv2_vis = tf.layers.conv2d(
                inputs=pool1_vis,
                filters=32,
                kernel_size=(3, 15),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv2_vis",
        )

        pool2_vis = tf.layers.max_pooling2d(
                inputs=conv2_vis, pool_size=(2, 2), strides=(2, 2), name="pool2_vis"
        )

        conv3_vis = tf.layers.conv2d(
                inputs=pool2_vis,
                filters=64,
                kernel_size=(3, 9),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv3_vis",
        )

        pool3_vis = tf.layers.max_pooling2d(
                inputs=conv3_vis, pool_size=(1, 2), strides=(1, 2), name="pool3_vis"
        )

        conv4_vis = tf.layers.conv2d(
                inputs=pool3_vis,
                filters=64,
                kernel_size=(3, 7),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv4_vis",
        )

        pool4_vis = tf.layers.max_pooling2d(
                inputs=conv4_vis, pool_size=(1, 2), strides=(1, 2), name="pool4_vis"
        )

        conv5_vis = tf.layers.conv2d(
                inputs=pool4_vis,
                filters=64,
                kernel_size=(3, 5),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv5_vis",
        )

        conv5_vis_out = tf.identity(conv5_vis, name='conv5_vis_out')

        # pool5_vis = tf.layers.max_pooling2d(
        #         inputs=conv5_vis, pool_size=(1, 2), strides=(1, 2), name="pool5_vis"
        # )
        #
        # conv6_vis = tf.layers.conv2d(
        #         inputs=pool5_vis,
        #         filters=64,
        #         kernel_size=(3, 5),
        #         padding="same",
        #         activation=tf.nn.relu,
        #         use_bias=True,
        #         kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #         name="conv6_vis",
        # )

    flatten_vol = tf.contrib.layers.flatten(inputs=conv3)
    if vis_views:
        flatten_vis = tf.contrib.layers.flatten(inputs=conv5_vis_out)
        flatten = tf.concat([flatten_vol, flatten_vis, scales], axis=1, name="flatten")
        # flatten = tf.concat([flatten_vis], axis=1, name="flatten")
    else:
        flatten = tf.concat([flatten_vol, scales], axis=1, name="flatten")

    # classification network
    dense1 = tf.layers.dense(
        inputs=flatten,
        units=512,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        use_bias=True,
        name="dense1",
    )

    bn_dense1 = tf.layers.batch_normalization(
        dense1, training=training, name="bn_dense1"
    )

    dropout_dense1 = tf.layers.dropout(
        bn_dense1, rate=0.5, training=training, name="dropout_dense1"
    )

    descriptor = tf.layers.dense(
        inputs=dropout_dense1,
        units=64,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        use_bias=True,
        name="descriptor",
    )

    bn_descriptor = tf.layers.batch_normalization(
        descriptor, training=training, name="bn_descriptor"
    )

    with tf.name_scope("OutputScope") as scope:
        tf.add(bn_descriptor, 0, name="descriptor_bn_read")
        tf.add(descriptor, 0, name="descriptor_read")

    if triplet > 0:
        y_true_label = tf.math.argmax(y_true, axis=1)
        loss_c = tf.identity(tf.contrib.losses.metric_learning.triplet_semihard_loss(y_true_label,
                                                                                     descriptor,
                                                                                     0.1),
                             name="loss_c")
    else:
        dropout_descriptor = tf.layers.dropout(
            bn_descriptor, rate=0.35, training=training, name="dropout_descriptor"
        )

        y_pred = tf.layers.dense(
            inputs=dropout_descriptor,
            units=n_classes,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            activation=None,
            use_bias=True,
            name="classes",
        )

        loss_c = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true),
            name="loss_c",
        )


    # reconstruction network
    dec_dense1 = tf.layers.dense(
        inputs=descriptor,
        units=8192,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        use_bias=True,
        name="dec_dense1",
    )

    reshape = tf.reshape(dec_dense1, (tf.shape(cnn_input)[0], 8, 8, 4, 32))

    dec_conv1 = tf.layers.conv3d_transpose(
        inputs=reshape,
        filters=32,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        name="dec_conv1",
    )

    dec_conv2 = tf.layers.conv3d_transpose(
        inputs=dec_conv1,
        filters=32,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        name="dec_conv2",
    )

    dec_reshape = tf.layers.conv3d_transpose(
        inputs=dec_conv2,
        filters=1,
        kernel_size=(3, 3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.sigmoid,
        name="dec_reshape",
    )

    reconstruction = dec_reshape
    with tf.name_scope("ReconstructionScopeAE") as scope:
        tf.add(reconstruction, 0, name="ae_reconstruction_read")

    FN_TO_FP_WEIGHT = 0.9
    loss_r = -tf.reduce_mean(
        FN_TO_FP_WEIGHT * cnn_input * tf.log(reconstruction + 1e-10)
        + (1 - FN_TO_FP_WEIGHT) * (1 - cnn_input) * tf.log(1 - reconstruction + 1e-10)
    )
    tf.identity(loss_r, "loss_r")

    # training
    LOSS_R_WEIGHT = 200
    LOSS_C_WEIGHT = 1
    loss = tf.add(LOSS_C_WEIGHT * loss_c, LOSS_R_WEIGHT * loss_r, name="loss")
    # loss = tf.add(LOSS_C_WEIGHT * loss_c, 0, name="loss")

    global_step = tf.Variable(0, trainable=False, name="global_step")
    update_step = tf.assign(
        global_step, tf.add(global_step, tf.constant(1)), name="update_step"
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    # add batch normalization updates to the training operation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, name="train_op")

    # statistics
    # y_prob = tf.nn.softmax(y_pred, name="y_prob")

    if triplet > 0:
        accuracy = tf.identity(loss_c, name="accuracy")
    else:
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    roc_auc = tf.placeholder(dtype=tf.float32, shape=(), name="roc_auc")

    img_heatmap = tf.placeholder(dtype=tf.uint8, shape=(None,) + (None, None) + (3,), name="img_heatmap")
    img_heatmap_vis = tf.placeholder(dtype=tf.uint8, shape=(None,) + input_shape_vis + (3,), name="img_heatmap_vis")
    score = tf.placeholder(dtype=tf.float32, shape=(None,), name="score")
    score_vis = tf.placeholder(dtype=tf.float32, shape=(None,), name="score_vis")

    with tf.name_scope("summary"):
        tf.summary.scalar("loss", loss, collections=["summary_batch"])
        tf.summary.scalar("loss_c", loss_c, collections=["summary_batch"])
        tf.summary.scalar("loss_r", loss_r, collections=["summary_batch"])
        tf.summary.scalar("accuracy", accuracy, collections=["summary_batch"])

        tf.summary.image('heatmap', img_heatmap, collections=["summary_heatmap"], max_outputs=8)
        tf.summary.image('heatmap_vis', img_heatmap_vis, collections=["summary_heatmap", "summary_heatmap_vis"], max_outputs=8)
        tf.summary.text('score', tf.strings.as_string(score), collections=["summary_heatmap"])
        tf.summary.text('score_vis', tf.strings.as_string(score_vis), collections=["summary_heatmap", "summary_heatmap_vis"])

        tf.summary.scalar("roc_auc", roc_auc, collections=["summary_epoch"])
