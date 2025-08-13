import jax
import jax.numpy as jnp

import flax.nnx as nnx
from flax.traverse_util import flatten_dict

import optax

from orbax import checkpoint as ocp

import hydra
from omegaconf import DictConfig, OmegaConf

import grain.python as grain

import mlflow

import os
from functools import partial
import random
from pathlib import Path
from tqdm import tqdm
from collections.abc import Callable

from DataSource import ImageDataSource
from utils import init_tx, initialize_dataloader


class L2DPopModel(nnx.Module):
    def __init__(
            self,
            base_model_fn: Callable,
            feature_dim: int,
            num_classes: int,
            embedding_dim: int,
            rngs: nnx.Rngs
    ) -> None:
        self.feature_extractor = base_model_fn(
            num_classes=feature_dim,
            rngs=rngs
        )
        self.clf = nnx.Linear(
            in_features=feature_dim,
            out_features=num_classes,
            rngs=rngs
        )
        self.embedding = nnx.Embed(
            num_embeddings=num_classes,
            features=embedding_dim,
            rngs=rngs
        )

        hidden_dim = (feature_dim + embedding_dim) // 2
        self.deferral_projection = nnx.Sequential(
            nnx.Linear(
                in_features=2 * feature_dim + embedding_dim,
                out_features=hidden_dim,
                rngs=rngs
            ),
            nnx.BatchNorm(num_features=hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                in_features=hidden_dim,
                out_features=1,
                rngs=rngs
            )
        )

    def __call__(
        self,
        x: jax.Array,
        human_representation: jax.Array  # (num_humans, feature_dim + embedding_dim)
    ) -> jax.Array:
        """perform a learning to defer following the Multi-L2D approach

        Args:
            x: input sample
            human_representation:  # (num_humans, feature_dim + embedding_dim)

        Returns:
            logits:
        """
        features = self.feature_extractor(x)  # (B, feature_dim)

        # classifier
        logits_clf = self.clf(features)  # (B, C)

        # region HUMAN
        human_repre = jnp.broadcast_to(
            array=human_representation[None, :, :],
            shape=(len(x), *human_representation.shape)
        )  # (B, num_humans, feature_dim + embedding_dim)

        features_broadcast = jnp.broadcast_to(
            array=features[:, None, :],
            shape=(
                len(features),
                human_representation.shape[0],
                features.shape[-1]
            )
        )  # (B, num_humans, feature_dim)

        features_human = jnp.concatenate(
            arrays=(features_broadcast, human_repre),
            axis=-1
        )  # (B, num_humans, 2 * feature_dim + embedding_dim)

        logits_human = self.deferral_projection(features_human)  # (B, num_humans, 1)
        # endregion

        logits = jnp.concatenate(
            arrays=(
                jnp.broadcast_to(
                    array=logits_clf[:, None, :],
                    shape=(len(logits_clf), logits_human.shape[1], logits_clf.shape[-1])
                ),
                logits_human
            ),
            axis=-1
        )  # (B, num_humans, C + 1)

        return logits

    def encode_annotator(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """encode the data in a set to a vector to represent a human annotator

        Args:
            x: input samples  # (B, d)
            y: annotations (in integer)  # (B, num_humans)

        Returns:
            psi: vector representing human annotator
                # (num_humans, feature_dim + embedding_dim)
        """
        features = self.feature_extractor(x)  # (B, feature_dim)
        annotation_embedding = self.embedding(y)  # (B, num_humans, embedding_dim)

        features = jnp.broadcast_to(
            array=features[:, None, :],
            shape=(len(features), y.shape[-1], features.shape[-1])
        )  # (B, num_humans, feature_dim)

        r = jnp.concatenate(
            arrays=(features, annotation_embedding),
            axis=-1
        )  # (B, num_humans, feature_dim + embedding_dim)

        # aggregate with deep set
        psi = jnp.mean(
            a=r,
            axis=0
        )  # (num_humans, feature_dim + embedding_dim)

        return psi


@partial(jax.jit, static_argnames=('num_classes',))
def augment_labels(y: jax.Array, t: jax.Array, num_classes: int) -> jax.Array:
    """augment the labels for the unified gating + classifier model

    Args:
        y: ground truth labels (batch,)
        t: expert's annotations (missing is denoted as -1) (batch, num_experts)
        num_classes:

    Return:
        y_augmented:
    """
    y_one_hot = jax.nn.one_hot(
        x=y,
        num_classes=num_classes
    )  # (batch, num_classes)

    # binary flag of expert's predictions
    y_orthogonal = (t == y[:, None]) * 1  # (batch, num_experts)

    y_broadcast = jnp.broadcast_to(
        array=y_one_hot[:, None, :],  # (batch, 1, num_classes)
        shape=(len(y), t.shape[1], num_classes)
    )  # (batch, num_experts, num_classes)

    y_augmented = jnp.concatenate(
        arrays=(
            y_broadcast,  # (batch, num_experts, num_classes)
            y_orthogonal[:, :, None]  # (batch, num_experts, 1)
        ),
        axis=-1
    )  # (batch, num_experts, num_classes + 1)

    return y_augmented


@partial(
    nnx.jit,
    static_argnames=(
        'num_classes',
        'dirichlet_concentration',
        'dataset_length'
    )
)
def loss_fn(
    model: L2DPopModel,
    x: jax.Array,
    y_augmented: jax.Array,
    x_ctx: jax.Array,
    t_ctx: jax.Array,
    num_classes: int,
    dirichlet_concentration: list[float],
    dataset_length: int
) -> jax.Array:
    """a wrapper to calculate the loss

    Args:
        model: the 'unified" model
        x: input samples
        y_augmented: the first num_classes elements corresponds to the
            ground truth label to train the classifier, while the remainings
            correspond to the correction of each human expert
        num_classes:
        dirichlet_concentration:
        dataset_length

    Returns:
        loss: the total loss including the prior as well
    """
    human_representation = model.encode_annotator(x=x_ctx, y=t_ctx)
    logits = model(x=x, human_representation=human_representation)

    # region DEFERRAL LOSS
    loss_defer = optax.losses.softmax_cross_entropy(
        logits=logits,
        labels=y_augmented
    )
    loss_defer = jnp.mean(a=loss_defer, axis=(0, 1))
    # endregion

    # region PRIOR
    # apply log-softmax on the logits 
    log_softmax = jax.nn.log_softmax(
        x=logits,
        axis=-1
    )  # (batch, num_experts, num_classes + 1)
    log_softmax_clf = jax.nn.logsumexp(
        a=log_softmax[:, :num_classes],
        axis=-1
    )  # (batch, num_experts)
    log_logits_gating = jnp.concatenate(
        arrays=(log_softmax[:, :, num_classes:], log_softmax_clf[:, :, None]),
        axis=-1
    )  # (batch, num_experts, 2)

    loss_prior = -jnp.sum(
        a=(jnp.array(object=dirichlet_concentration) - 1) * log_logits_gating,
        axis=-1
    )  # (batch, num_experts)
    loss_prior = jnp.mean(a=loss_prior, axis=(0, 1))
    # endregion

    loss = loss_defer + (len(x) / dataset_length) * loss_prior

    return loss


@partial(
    nnx.jit,
    static_argnames=(
        'num_classes',
        'dirichlet_concentration',
        'dataset_length'
    )
)
def train_step(
    x: jax.Array,  # (batch, d)
    y_augmented: jax.Array,  # (batch, num_experts, num_classes + 1)
    x_ctx: jax.Array,
    t_ctx: jax.Array,  # (batch, num_experts),
    model: L2DPopModel,
    optimizer: nnx.Optimizer,
    num_classes: int,
    dirichlet_concentration: list[float],
    dataset_length: int
) -> tuple[L2DPopModel, nnx.Optimizer, jax.Array]:
    """
    """
    grad_value_fn = nnx.value_and_grad(f=loss_fn, argnums=0)
    loss, grads = grad_value_fn(
        model,
        x,
        y_augmented,
        x_ctx,
        t_ctx,
        num_classes,
        dirichlet_concentration,
        dataset_length
    )

    optimizer.update(model=model, grads=grads)

    return (model, optimizer, loss)


def train(
    dataloader: grain.DatasetIterator,
    model: L2DPopModel,
    optimizer: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[L2DPopModel, nnx.Optimizer, jax.Array]:
    """
    """
    # metric to track the training loss
    loss_accum = nnx.metrics.Average()

    model.train()

    for _ in tqdm(
        iterable=range(cfg.dataset.length.train // cfg.training.batch_size),
        desc='epoch',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        samples = next(dataloader)

        x = jnp.asarray(
            a=samples['image'],
            dtype=eval(cfg.jax.dtype)
        )  # input samples
        y = jnp.asarray(
            a=samples['ground_truth'],
            dtype=jnp.int32
        )  # true int labels  (batch,)
        t = jnp.asarray(
            a=samples['label'],
            dtype=jnp.int32
        )  # annotated int labels (batch, num_experts)

        # augmented labels
        y_augmented = augment_labels(
            y=y[len(y)//2:],
            t=t[len(y)//2:],
            num_classes=cfg.dataset.num_classes
        )

        model, optimizer, loss = train_step(
            x=x[len(x) // 2:],
            y_augmented=y_augmented,
            x_ctx=x[:len(x) // 2],
            t_ctx=t[:len(t) // 2],
            model=model,
            optimizer=optimizer,
            num_classes=cfg.dataset.num_classes,
            dirichlet_concentration=cfg.hparams.dirichlet_concentration,
            dataset_length=cfg.dataset.length.train
        )

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        # tracking
        loss_accum.update(values=loss)

    return model, optimizer, loss_accum.compute()


def evaluate(
    dataloader: grain.DataLoader,
    human_representations: jax.Array,
    model: L2DPopModel,
    cfg: DictConfig
) -> tuple[list[jax.Array], list[jax.Array], jax.Array]:
    """
    """
    accuracy_accums = [nnx.metrics.Accuracy() for _ in range(len(human_representations))]
    coverages = [nnx.metrics.Average() for _ in range(len(human_representations))]
    clf_accuracy_accum = nnx.metrics.Accuracy()

    model.eval()

    for samples in tqdm(
        iterable=dataloader,
        total=cfg.dataset.length.test//cfg.training.batch_size + 1,
        desc='evaluate',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.asarray(
            a=samples['image'],
            dtype=jnp.float32
        )  # input samples
        y = jnp.asarray(
            a=samples['ground_truth'],
            dtype=jnp.int32
        )  # true labels (batch,)
        t = jnp.asarray(
            a=samples['label'],
            dtype=jnp.int32
        )  # annotated labels (batch, num_experts)

        logits = model(
            x=x,
            human_representation=human_representations
        )  # (batch, num_experts, num_classes + 1)

        # classifier predictions
        clf_logits = logits[:, :, :cfg.dataset.num_classes]
        clf_logits = jnp.mean(a=clf_logits, axis=1)  # (batch, num_classes)
        clf_predictions = jnp.argmax(
            a=clf_logits,
            axis=-1
        )  # (batch,)

        clf_accuracy_accum.update(
            logits=clf_logits,
            labels=y
        )

        labels_concatenated = jnp.concatenate(
            arrays=(
                jnp.broadcast_to(
                    array=clf_predictions[:, None, None],
                    shape=(len(clf_predictions), t.shape[1], 1)
                ),
                t[:, :, None],
            ),
            axis=-1
        )  # (batch, num_experts, 2)

        logits_max_id = jnp.argmax(a=logits, axis=-1)  # (batch, num_experts)
        logits_max_id = logits_max_id - cfg.dataset.num_classes

        # which samples are predicted by classifier
        samples_predicted_by_clf = (logits_max_id < 0) * 1  # (batch, num_experts)

        for i in range(len(coverages)):
            coverages[i].update(values=samples_predicted_by_clf[:, i])

        # system's predictions
        y_predicted = jnp.take_along_axis(
            arr=labels_concatenated,
            indices=1 - samples_predicted_by_clf[:, :, None],
            axis=-1
        )  # (batch, num_experts, 1)
        y_predicted = jnp.squeeze(a=y_predicted)  # (batch, num_experts)

        for i in range(len(accuracy_accums)):
            accuracy_accums[i].update(
                logits=jax.nn.one_hot(
                    x=y_predicted[:, i],
                    num_classes=cfg.dataset.num_classes
                ),
                labels=y
            )

    return (
        [accuracy_accum.compute() for accuracy_accum in accuracy_accums],
        [coverage.compute() for coverage in coverages],
        clf_accuracy_accum.compute()
    )


@hydra.main(version_base=None, config_path='conf', config_name='conf')
def main(cfg: DictConfig) -> None:
    """main procedure
    """
    # region ENVIRONMENT
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)
    # endregion

    # region DATASETS
    source_train = ImageDataSource(
        annotation_files=cfg.dataset.train_files,
        ground_truth_file=cfg.dataset.train_ground_truth_file,
        root=cfg.dataset.root
    )

    source_test = ImageDataSource(
        annotation_files=cfg.dataset.test_files,
        ground_truth_file=cfg.dataset.test_ground_truth_file,
        root=cfg.dataset.root
    )

    OmegaConf.set_struct(conf=cfg, value=True)
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.train',
        value=len(source_train),
        force_add=True
    )
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.test',
        value=len(source_test),
        force_add=True
    )
    # endregion

    model = L2DPopModel(
        base_model_fn=partial(
            hydra.utils.instantiate(config=cfg.model),
            dropout_rate=cfg.training.dropout_rate,
            dtype=eval(cfg.jax.dtype)
        ),
        feature_dim=128,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=128,
        rngs=nnx.Rngs(jax.random.PRNGKey(seed=random.randint(a=0, b=100)))
    )
    optimizer = nnx.Optimizer(
        model=model,
        tx=init_tx(
            dataset_length=len(source_train),
            lr=cfg.training.lr,
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
            clipped_norm=cfg.training.clipped_norm,
            key=random.randint(a=0, b=100)
        ),
        wrt=nnx.Param
    )

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=50,
        max_to_keep=10,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )

    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(
            run_id=cfg.experiment.run_id,
            log_system_metrics=False) as mlflow_run, \
        ocp.CheckpointManager(
            directory=os.path.join(
                os.getcwd(),
                cfg.experiment.logdir,
                cfg.experiment.name,
                mlflow_run.info.run_id
            ),
            options=ckpt_options) as ckpt_mngr:

        if cfg.experiment.run_id is None:
            start_epoch_id = 0

            # log hyper-parameters
            mlflow.log_params(
                params=flatten_dict(
                    xs=OmegaConf.to_container(cfg=cfg), sep='.'
                )
            )

            # log source code
            mlflow.log_artifact(
                local_path=os.path.abspath(path=__file__),
                artifact_path='source_code'
            )
        else:
            start_epoch_id = ckpt_mngr.latest_step()

            checkpoint = ckpt_mngr.restore(
                step=start_epoch_id,
                args=ocp.args.StandardRestore(item=nnx.state(model))
            )

            nnx.update(model, checkpoint)

            del checkpoint

    # create iterative datasets as data loaders
        dataloader_train = initialize_dataloader(
            data_source=source_train,
            num_epochs=cfg.training.num_epochs - start_epoch_id + 1,
            shuffle=True,
            seed=random.randint(a=0, b=255),
            batch_size=2 * cfg.training.batch_size,
            resize=cfg.data_augmentation.resize,
            padding_px=cfg.data_augmentation.padding_px,
            crop_size=cfg.data_augmentation.crop_size,
            mean=cfg.data_augmentation.mean,
            std=cfg.data_augmentation.std,
            p_flip=cfg.data_augmentation.prob_random_flip,
            num_workers=cfg.data_loading.num_workers,
            num_threads=cfg.data_loading.num_threads,
            prefetch_size=cfg.data_loading.prefetch_size
        )
        dataloader_train = iter(dataloader_train)

        dataloader_ctx_fn = partial(
            initialize_dataloader,
            data_source=source_train,
            num_epochs=1,
            shuffle=True,
            batch_size=cfg.training.batch_size,
            resize=cfg.data_augmentation.resize,
            padding_px=cfg.data_augmentation.padding_px,
            crop_size=cfg.data_augmentation.crop_size,
            mean=cfg.data_augmentation.mean,
            std=cfg.data_augmentation.std,
            p_flip=cfg.data_augmentation.prob_random_flip,
            num_workers=cfg.data_loading.num_workers,
            num_threads=cfg.data_loading.num_threads,
            prefetch_size=cfg.data_loading.prefetch_size
        )

        dataloader_test = initialize_dataloader(
            data_source=source_test,
            num_epochs=1,
            shuffle=False,
            seed=0,
            batch_size=cfg.training.batch_size,
            resize=cfg.data_augmentation.crop_size,
            padding_px=None,
            crop_size=None,
            mean=cfg.data_augmentation.mean,
            std=cfg.data_augmentation.std,
            p_flip=None,
            is_color_img=True,
            num_workers=cfg.data_loading.num_workers,
            num_threads=cfg.data_loading.num_threads,
            prefetch_size=cfg.data_loading.prefetch_size
        )

        for epoch_id in tqdm(
            iterable=range(start_epoch_id, cfg.training.num_epochs, 1),
            desc='progress',
            ncols=80,
            leave=True,
            position=1,
            colour='green',
            disable=not cfg.data_loading.progress_bar
        ):
            model, optimizer, loss = train(
                dataloader=dataloader_train,
                model=model,
                optimizer=optimizer,
                cfg=cfg
            )

            mlflow.log_metric(
                key='loss',
                value=loss.item(),
                step=epoch_id + 1,
                synchronous=False
            )

            if (epoch_id + 1) % cfg.training.eval_every_n_epochs == 0:
                dataloader_ctx = iter(dataloader_ctx_fn(
                    seed=random.randint(a=0, b=255)
                ))
                samples = next(dataloader_ctx)
                x_ctx = jnp.array(
                    object=samples['image'],
                    dtype=eval(cfg.jax.dtype)
                )
                t_ctx = jnp.array(
                    object=samples['label'],
                    dtype=jnp.int32
                )

                model.eval()
                human_representations = model.encode_annotator(
                    x=x_ctx,
                    y=t_ctx
                )

                accuracies, coverages, clf_accuracy = evaluate(
                    dataloader=dataloader_test,
                    human_representations=human_representations,
                    model=model,
                    cfg=cfg
                )

                mlflow.log_metric(
                    key='accuracy/clf',
                    value=clf_accuracy.item(),
                    step=epoch_id + 1,
                    synchronous=False
                )
                mlflow.log_metrics(
                    metrics={
                        f'human_{i}/accuracy': accuracies[i].item() \
                            for i in range(len(accuracies))
                    },
                    step=epoch_id + 1,
                    synchronous=False
                )
                mlflow.log_metrics(
                    metrics={
                        f'human_{i}/coverage': coverages[i].item() \
                            for i in range(len(coverages))
                    },
                    step=epoch_id + 1,
                    synchronous=False
                )

            # wait for checkpoint manager completing the asynchronous saving
            # before saving another checkpoint
            ckpt_mngr.wait_until_finished()

            # save parameters asynchronously
            ckpt_mngr.save(
                step=epoch_id + 1,
                args=ocp.args.StandardSave(nnx.state(model))
            )


if __name__ == '__main__':
    main()
