import os, logging, multiprocessing, re
log = logging.getLogger(__name__)
import hydra
import pkgmgr as opentf
def init_process(): logging.basicConfig(level=logging.INFO)

def __(fpred, adila, minorities, ratios, algorithm, k_max, alpha, evalcfg):
    preds, preds_, fpred_ = adila.rerank(fpred, minorities, ratios, algorithm, k_max, alpha)
    adila.eval_fair(preds, minorities, preds_, fpred_, ratios, k_max, evalcfg.metrics.fair, evalcfg.per_instance)
    adila.eval_utility(preds, fpred, preds_, fpred_, k_max, evalcfg.metrics, evalcfg.per_instance)
    return fpred_

def _(adila, fpred, minorities, ratios, algorithm, k_max, alpha, acceleration, evalcfg):
    outputs = []
    if os.path.isfile(fpred): outputs.append(__(fpred, adila, minorities, ratios, algorithm, k_max, alpha, evalcfg))
    elif os.path.isdir(fpred):
        log.info(f'Queuing all *.pred files at {fpred} for {opentf.textcolor["cyan"]}{adila} ... {opentf.textcolor["reset"]}');
        import glob; from functools import partial
        fpreds = glob.glob(f'{glob.escape(fpred)}/f*.test.*pred')
        if 'per_epoch' not in evalcfg or not evalcfg.per_epoch: fpreds = [f for f in fpreds if not re.search(r'\.e\d+\.', os.path.basename(f))]
        if not fpreds: log.info(f'{opentf.textcolor["yellow"]}Nothing found! {opentf.textcolor["reset"]}'); return;

        adila.n_processes = multiprocessing.cpu_count() - 1 if acceleration == 'cpu' else int(acceleration.split(':')[1])
        if adila.n_processes < 2:
            for fpred in fpreds: outputs.append(__(fpred, adila, minorities, ratios, algorithm, k_max, alpha, evalcfg))
        else:
            with multiprocessing.Pool(initializer=init_process, processes=min(adila.n_processes, len(fpreds))) as p:
                outputs = p.map(partial(__, adila=adila, minorities=minorities, ratios=ratios, algorithm=algorithm, k_max=k_max, alpha=alpha, evalcfg=evalcfg), fpreds)
    return sorted(outputs)

@hydra.main(version_base=None, config_path='.', config_name='__config__')
def run(cfg) -> None:
    from adila import Adila
    adila = Adila(cfg.data.fteamsvecs, cfg.data.fsplits, cfg.data.fgender)
    for key in cfg.eval.metrics:
        if key != 'topk': cfg.eval.metrics[key] = [m.replace('topk', cfg.eval.metrics.topk) for m in cfg.eval.metrics[key]]
    for attribute in cfg.fair.attribute:
        for notion in cfg.fair.notion:
            for is_popular_alg in [None] if attribute == 'gender' else cfg.fair.is_popular_alg:  # is_popular_alg has nothing to do with gender debiasing, so single default value
                stats, minorities, ratios = adila.prep(cfg.data.output, notion, attribute, is_popular_alg, cfg.fair.is_popular_coef)
                if cfg.fair.manual_ratio: ratios = [cfg.fair.manual_ratio] * len(ratios)
                for algorithm in cfg.fair.algorithm:
                    _(adila, cfg.data.fpred, minorities, ratios, algorithm, cfg.fair.k_max, cfg.fair.alpha, cfg.acceleration, cfg.eval)

if __name__ == '__main__': run()