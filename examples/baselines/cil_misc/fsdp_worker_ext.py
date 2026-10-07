"""fsdp_worker_ext.py
===============================================================================
Worker-base machinery for the persistent-worker CIL trainers.

``PersistentRefFSDPWorker`` (defined in each CIL entry point) subclasses verl's
``FSDPWorker`` to add a frozen *anchor* (previous-task) model and in-memory
actor→anchor weight copies. The reference module stays at the original
pretrained weights. It must remain compatible with verl's colocated-worker
construction:

``create_colocated_worker_cls`` reads ``worker_cls.__base__`` and calls
``super().__init__()`` with *no arguments*, which only works when ``__base__``
resolves to ``Worker`` (whose ``__init__`` accepts no required args).  A direct
``FSDPWorker`` subclass would set ``__base__ = FSDPWorker`` and crash, because
``FSDPWorker.__init__`` requires ``config`` and ``role``.

``_FSDPWorkerColocBase`` is a no-op proxy inserted as the *first* parent so that
``__base__`` points back to ``Worker`` while the MRO still routes
``super().__init__(config, role)`` through to ``FSDPWorker``.
"""

from verl.single_controller.base.worker import Worker


class _FSDPWorkerColocBase(Worker):
    """Transparent proxy base for colocated ``FSDPWorker`` subclasses.

    ``create_colocated_worker_cls`` in verl reads
    ``worker_cls = UserClass.__base__`` and inherits ``WorkerDict`` from it,
    then calls ``super().__init__()`` with *no arguments*.  For all built-in
    verl workers the chain is ``ConcreteWorker -> Worker``, so
    ``Worker.__init__(cuda_visible_devices=None)`` is safely called.

    ``PersistentRefFSDPWorker`` extends ``FSDPWorker``, which would make its
    ``__base__`` ``FSDPWorker`` instead of ``Worker``.  ``FSDPWorker.__init__``
    requires ``config`` and ``role``, causing a ``TypeError`` at startup.

    Inserting this no-op class as the *first* parent shifts ``__base__`` back
    to this proxy (whose own ``__base__`` is ``Worker``), and the MRO still
    routes ``super().__init__(config, role)`` through to ``FSDPWorker``.
    """
    # No __init__: inherits Worker.__init__(self, cuda_visible_devices=None)
