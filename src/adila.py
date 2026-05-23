import os, pickle, logging, multiprocessing
log = logging.getLogger(__name__)

import pkgmgr as opentf

pd = opentf.install_import('pandas')
tqdm = opentf.install_import('tqdm', from_module='tqdm')
torch = opentf.install_import('torch')

class Adila:

    def __init__(self, fteamsvecs, fsplits, fgender, n_processes=1):
        if isinstance(fteamsvecs, dict): self.teamsvecs = fteamsvecs
        else:
            with open(fteamsvecs, 'rb') as f: self.teamsvecs = pickle.load(f)
        if isinstance(fsplits, dict): self.splits = fsplits
        else:
            with open(fsplits, 'rb') as f: self.splits = pickle.load(f)

        self.attribute = None
        self.fair_notion = None
        self.is_popular_alg = None
        self.fgender = fgender
        self.minorities = []
        self.n_processes = n_processes

    def __str__(self): return f'{self.attribute}.{self.fair_notion}.{self.is_popular_alg}'
    def _get_labeled_sorted_preds(self, preds, minorities, k_max):
        if not preds.is_sparse: sorted_probs, sorted_indices = preds.sort(dim=1, descending=True)  # |Test| * |Experts|
        else: #|Test| * |topK == k_max|, we need to avoid working with dense
            preds = preds.coalesce()
            rows, cols = preds.indices()
            vals = preds.values()
            order = torch.argsort(rows * (vals.max() + 1) - vals) # row-wise descending sort
            rows, cols, vals = rows[order], cols[order], vals[order]
            # print(torch.bincount(rows))
            splits = torch.split(cols, torch.bincount(rows).tolist())
            probs_splits = torch.split(vals, torch.bincount(rows).tolist())
            # pad each row to k_max (or preds.size(1) but that would be dense again)
            sorted_indices = torch.stack([torch.cat([x, torch.tensor([i for i in range(k_max) if i not in x])[:(k_max - len(x))]]) if len(x) < k_max else x[:k_max] for x in splits]) # pad col idx of zero values
            sorted_probs   = torch.stack([torch.cat([x, x.new_zeros(k_max - len(x))]) if len(x) < k_max else x[:k_max] for x in probs_splits]) # pad zero values

        sorted_labels = (sorted_indices[..., None] == torch.tensor(minorities)).any(dim=-1)
        ## if |experts| are small/mid scale >> dense vector of boolean labels
        # labels = torch.zeros(preds.shape[1], dtype=torch.bool, device=preds.device)
        # labels[minorities] = True
        # sorted_labels = labels[sorted_indices]  # torch uses advanced indexing, not broadcasting! still |Test| * |Experts|
        return torch.stack([sorted_indices, sorted_labels.to(sorted_indices.dtype), sorted_probs], dim=-1)
        # [[expertid, minority label, ranked prob], ...]

    def prep(self, output, fair_notion='dp', attribute='popularity', is_popular_alg='avg', coef=1.0) -> tuple: #coefficient to calculate a threshold for popularity (e.g. if 1.5, threshold = 1.5 * average number of teams per expert)
        self.output = f'{output}/adila/{attribute}{"." + is_popular_alg if attribute == "popularity" else ""}'
        if not os.path.isdir(self.output): os.makedirs(self.output)
        if not os.path.isdir(f'{self.output}/{fair_notion}'): os.makedirs(f'{self.output}/{fair_notion}')
        self.attribute = attribute
        self.fair_notion = fair_notion
        self.is_popular_alg = is_popular_alg

        try:
            log.info(f'Loading stats, ratios, and ids for minority experts ...')
            with open(f'{self.output}/stats.pkl', 'rb') as f: stats = pickle.load(f)
            minorities = pd.read_csv(f'{self.output}/labels.csv').iloc[:, 0].tolist()
            if self.fair_notion == 'eo':
                with open(f'{self.output}/eo/ratios.pkl', 'rb') as f: ratios = pickle.load(f)
        except (FileNotFoundError, EOFError):
            log.info(f'Loading failed! Generating files at {self.output} ...')
            stats = {}
            stats['*nexperts'] = self.teamsvecs['member'].shape[1]
            col_sums = self.teamsvecs['member'].sum(axis=0)

            stats['nteams_expert-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}

            # many nonpopular/male but few popular/female. So, we only keep popular/female idxes as minorities.
            # this should be the same for all baselines, so read once from the file at ./output/{domain}/{dataset}
            if self.attribute == 'popularity':
                stats['*avg_nteams_expert'] = col_sums.mean()
                x, y = zip(*enumerate(sorted(col_sums.A1.astype(int), reverse=True)))
                if self.is_popular_alg == 'auc': import plot; stats['*auc_nteams_expert'] = plot.area_under_curve(x, y, 'expert-idx', 'nteams', show_plot=False)
                threshold = coef * stats[f'*{self.is_popular_alg}_nteams_expert']
                minorities = [expertidx for expertidx, nteam_expert in enumerate(col_sums.getA1()) if threshold <= nteam_expert] #rowid maps to columnid in teamvecs['member']
            elif self.attribute == 'gender': minorities = pd.read_csv(self.fgender).iloc[:, 0].tolist()
            stats['minority_ratio'] = len(minorities) / stats['*nexperts']
            with open(f'{self.output}/stats.pkl', 'wb') as f: pickle.dump(stats, f)
            pd.DataFrame(data=minorities, columns=['teamsvecs-experts-colidx']).to_csv(f'{self.output}/labels.csv', index=False)

            ratios = list()
            if self.fair_notion == 'eo': # we need to know per team's ratio of minorities
                skill_member = self.teamsvecs['skill'].transpose() @ self.teamsvecs['member']
                log.info(f'Generating ratios ... ')
                for i in tqdm(self.splits['test']):
                    team_skills = self.teamsvecs['skill'][i].nonzero()[1].tolist()
                    experts = [skill_member[idx].nonzero()[1] for idx in team_skills]
                    skill_holders = set(experts[0]).union(*experts)
                    assert skill_holders, f'{opentf.textcolor["red"]}No expert has team {i}\'s skills {team_skills}!{opentf.textcolor["reset"]}'
                    skill_holders_minorities = set(minorities).intersection(skill_holders)
                    ratios.append(len(skill_holders_minorities) / len(skill_holders))
                    with open(f'{self.output}/eo/ratios.pkl', 'wb') as file: pickle.dump(ratios, file)

        if self.fair_notion == 'dp': ratios = [stats['minority_ratio']]
        return stats, minorities, ratios

    def rerank(self, fpred, minorities, ratios, algorithm='det_greedy', k_max=100, alpha=0.05) -> tuple:
        """
        Args:
            fpred: the filename for predictions for test teams |test| * |experts|
            minorities: list of expert-idx who are minorities like females or populars
            ratios: desired ratio of protected experts in the output
            algorithm: ranker algorithm of choice among {'det_greedy', 'det_cons', 'det_relaxed', 'fa-ir'}
            k_max: maximum number of returned team members by reranker
            alpha: significance value for fa*ir algorithm
        Returns:
            preds: loaded predictions (probs) |test| * |experts|
            preds_: adjusted predictions (probs) after reranking |test| * |experts|
            fpred_: the filename for the saved reranked_preds
        """
        preds = torch.load(fpred, map_location='cpu')['y_pred']
        log.info(f'Reranking {fpred} using {opentf.textcolor["blue"]}{algorithm} with {k_max} cutoff ...{opentf.textcolor["reset"]}')
        # preds = torch.tensor([[0.1, 0.5, 0.3, 0.4,  0.1, 0.8, 0.3]])
        fpred_ = f'{self.output}/{self.fair_notion}/{os.path.split(fpred)[-1]}.{algorithm}.{self.is_popular_alg + "." if self.attribute=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "") + "." if algorithm=="fa-ir" else ""}{k_max}.rerank.pred'
        try:
            log.info(f'Loading reranked file {fpred_} for {fpred} if exists ...')
            with open(fpred_, 'rb') as f: preds_ = pickle.load(f)
        except FileNotFoundError:
            log.info(f'No existing rerank version. Reranking {fpred} ...')
            # start_time = perf_counter()
            r = min(max(ratios[0], 0.1), 0.9) #clamps to stay between [0.1,0.9]

            if algorithm == 'fa-ir':
                fsc = opentf.install_import('fairsearchcore')
                fair = fsc.Fair(min(k_max, preds.shape[1]), 1 - r if self.attribute == 'popularity' else r, alpha) #r: proportion of protected candidates (gender, or 1 - popular for nonpopular) in the topK elements (should be between 0.02 and 0.98)
            elif algorithm in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort']:
                frr = opentf.install_import('reranking')

            preds_ = preds.detach().clone() #for the final reranked probs
            teams_ = self._get_labeled_sorted_preds(preds, minorities, k_max)
            # [[expertid, minority label, ranked prob], ...] ==> up to k_max tuples

            for i, team_ in enumerate(tqdm(teams_)):
                if self.fair_notion == 'eo': r = min(max(ratios[i], 0.1), 0.9)  # dynamic ratio r, clamps to stay between [0.1,0.9]
                if algorithm == 'fa-ir':
                    # FairScoreDocs needs True label for the members of the protected group.
                    # For gender, our minorities and protected group is the same, i.e., females.
                    # For popularilty, our minorities are populars but the protected group is non-populars. So, 'not' of their minority labels
                    experts = [fsc.models.FairScoreDoc(int(m[0]), float(m[2]), not bool(m[1]) if self.attribute=='popularity' else bool(m[1])) for m in team_]
                    # Reset the Fair obj to dynamic ratio r
                    if self.fair_notion == 'eo': fair = fsc.Fair(min(k_max, preds.shape[1]), 1 - r if self.attribute == 'popularity' else r, alpha)  # fair.p = r; fair._cache = {} #reset the Fair obj but it's buggy

                    # fairsearchcore/fail_prob.py L#177 in __hash__(), cast to int. The value of self.remaining_candidates is of numpy type!
                    # see https://github.com/fair-search/fairsearch-fair-python/issues/4
                    if fair.is_fair(experts[:k_max]): experts_ = experts[:k_max] #no change
                    else: experts_ = fair.re_rank(experts)[:k_max]
                    experts_ = [x.id for x in experts_]
                    # reranked_idx = [2, 0, 1, 5, 4, 3, 6]

                elif algorithm in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort']:
                    experts_ = frr.rerank([bool(label) for _, label, _ in team_], {True: r, False: 1 - r}, None, min(k_max, preds.shape[1]), algorithm, verbose=False) #verbose=True, a dataframe with more info
                    # reranked_idx = [2, 0, 1, 5, 4, 3, 6]

                # elif algorithm == 'fair_greedy':
                #     #TODO refactor and parameterize this algorithm
                #     bias_dict = dict([(member_probs.index(m), {'att': m[1], 'prob': m[2], 'idx': m[0]}) for m in member_probs[:500]])
                #     method = 'move_down'
                #     reranked_idx = fairness_greedy(bias_dict, r, 'att', method)[:k_max]
                #     reranked_probs = [bias_dict[idx]['prob'] for idx in reranked_idx[:k_max]]

                else: raise ValueError('Invalid fair reranking algorithm!')

                for j, expert_ in enumerate(experts_):
                    if not preds.is_sparse: preds_[i][expert_] = team_[j][2]
                    else: #sparse coo is immutable, cannot assign/modify values as it changes sparsity pattern
                        mask = (preds_.indices()[0] == i) & (preds_.indices()[1] == expert_) # within the k_max/topK nnz values
                        preds_.values()[mask] = team_[j][2]
                # we switch the top-rank probs for top-re-ranked experts
                # this way both lists give correct top experts after final rankings for evaluation
                # example:
                # preds: [0.1, 0.5, 0.3, 0.4, 0.1, 0.8, 0.3]
                # sorted preds: [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1] -> [5, 1, 3, 6, 2, 0, 4]
                # rerank: [2, 0, 1, 5, 4, 3, 6] -> assign top probs [0.5, 0.4, 0.8, 0.3, 0.3, 0.1, 0.1]
                # sorted rerank: [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1] -> [2, 0, 1, 5, 4, 3, 6]

            with open(fpred_, 'wb') as f: pickle.dump(preds_, f)
        return preds, preds_, fpred_

    def eval_fair(self, preds, minorities, preds_, fpred_, ratios, k_max, metrics=['skew', 'ndkl'], per_instance=False):
        """
        Args:
            preds: loaded predictions from a .pred file
            minorities: list of popular or female labels (true labels)
            preds_, fpred_: re-ranked probs considering a cut-off min(k_max, preds.shape[1]) and the stored filename
            ratios: inferred or a desired ratio of minorities
            k_max: cutoff for fair reranking methods
            metrics: fairness evaluation metrics
            per_instance: evaluation metric value for each test team instance
        Returns:
            None but the results are stored in *.csv files
        """
        log.info(f'{opentf.textcolor["green"]}Fairness evaluation for {fpred_} using {metrics} with {k_max} cutoff ...{opentf.textcolor["reset"]}')
        frr = opentf.install_import('reranking') # for ndkl and skew
        teams = self._get_labeled_sorted_preds(preds, minorities, k_max)  # [5, 1, 3, 6, 2, 0, 4] -> [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1]
        teams_ = self._get_labeled_sorted_preds(preds_, minorities, k_max)  # [2, 0, 1, 5, 4, 3, 6] -> [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1]

        results = []
        k_max = min(k_max, preds.shape[1])
        for i, (team, team_) in enumerate(tqdm(zip(teams, teams_))):
            lsteam, lsteam_ = team[:, 1][:k_max].bool().tolist(), team_[:, 1][:k_max].bool().tolist()
            if self.fair_notion == 'eo': r = min(max(ratios[i], 0.1), 0.9)  # dynamic ratio r, clamps to stay between [0.1,0.9]
            else: r = ratios[0]

            result = {}
            for metric in metrics:
                if 'ndkl' in metric:
                    topks = metric.split('_')[1].split(',')
                    for topk in map(int, topks):
                        result[f'before.ndkl_{topk}'] = frr.ndkl(lsteam[:topk], {True: r, False: 1 - r})
                        result[f'after.ndkl_{topk}'] = frr.ndkl(lsteam_[:topk], {True: r, False: 1 - r})

                if 'skew' in metric:
                    topks = metric.split('_')[1].split(',')
                    for topk in map(int, topks):
                        result[f'before.skew_{topk}.minority'] = frr.skew(lsteam[:topk].count(True)/topk, r)
                        result[f'before.skew_{topk}.majority'] = frr.skew(lsteam[:topk].count(False)/topk, 1 - r)
                        result[f'after.skew_{topk}.minority'] = frr.skew(lsteam_[:topk].count(True)/topk, r)
                        result[f'after.skew_{topk}.majority'] = frr.skew(lsteam_[:topk].count(False)/topk, 1 - r)

                # if metric in ['exp', 'expu']:
                #     frt = opentf.install_import('FairRankTune') #python 3.9+
                #     if metric == 'exp': exp_before, per_group_exp_before = frt.Metrics.EXP(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), 'MinMaxRatio')
                #     elif metric == 'expu': exp_before, per_group_exp_before = frt.Metrics.EXPU(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), pd.DataFrame(data=[j[2] for j in member_probs[:k_max]]),'MinMaxRatio')
                #
                #     try: before[metric]['protected'].append(per_group_exp_before[False])
                #     except KeyError: before[metric]['protected'].append(0)
                #     try: before[metric]['nonprotected'].append(per_group_exp_before[True])
                #     except KeyError: before[metric]['nonprotected'].append(0)
                #     before[metric][metric] = exp_before
                #
                #     if metric == 'exp': exp_after, per_group_exp_after = frt.Metrics.EXP(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), 'MinMaxRatio')
                #         # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                #         # dic_after[metric][metric] = exp_after
                #     elif metric == 'expu': exp_after, per_group_exp_after = frt.Metrics.EXPU(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), pd.DataFrame(data=[j[2] for i in reranked_idx[i][:k_max] for j in member_probs if j[0] == i]), 'MinMaxRatio')
                #         # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                #         # dic_after[metric][metric] = exp_after
                #
                #     try: after[metric]['protected'].append(per_group_exp_after[False])
                #     except KeyError: after[metric]['protected'].append(0)
                #     try:  after[metric]['nonprotected'].append(per_group_exp_after[True])
                #     except KeyError:  after[metric]['nonprotected'].append(0)
                #     after[metric][metric] = exp_after
            results.append(result)
        df = pd.DataFrame(results)
        if per_instance: df.to_csv(f'{fpred_}.eval.fair.instance.csv', index=False)

        df_mean = df.mean(axis=0).rename_axis('metrics').reset_index()
        df_mean[['type', 'metrics']] = df_mean['metrics'].str.split('.', n=1, expand=True)
        df_mean = df_mean.pivot_table(index='metrics', columns='type', values=0, sort=False).reset_index()
        df_mean.rename(columns={'before': 'mean.before', 'after': 'mean.after'}).to_csv(f'{fpred_}.eval.fair.mean.csv', index=False)
        log.info(f'Saved at {fpred_}.eval.fair.mean{"/instance" if per_instance else ""}.csv.')

    def eval_utility(self, preds, fpred, preds_, fpred_, k_max, metrics, per_instance=False) -> None:
        """
        Args:
            preds: the file for the predictions, *.pred file
            preds_: the file for the re-ranked probs considering a cut-off min(k_max, preds.shape[1]) and the stored filename
            k_max: cutoff for fair reranking methods
            metrics: utility evaluation metrics
            per_instance: evaluation metric value for each test team instance
        Returns:
            None but the results are stored in *.csv files
        """

        def _evaluate(Y_, metrics, per_instance, k_max):
            df, df_mean = pd.DataFrame(), pd.DataFrame()
            if not (metrics.trec or metrics.other): df, df_mean
            # evl = opentf.install_import('evl.metric', 'metric_')
            evl = opentf.install_import('evl.metric', 'evl.metric')
            # evl.metric works on numpy or scipy.sparse. so, we need to convert Y_ which is torch.tensor, either sparse or not
            Y_ = opentf.torch_sparse_2_scipy_sparse(Y_, 'csr') if Y_.is_sparse else Y_.cpu().numpy()
            # from https://github.com/fani-lab/OpeNTF/blob/main/src/mdl/ntf.py#L59
            if metrics.trec:
                log.info(f'{metrics.trec} ...')
                df, df_mean = evl.calculate_metrics(Y, Y_, k_max, per_instance, metrics.trec)
            if (m := [m for m in metrics.other if 'aucroc' in m]):
                log.info(f'{m} ...')
                aucroc, _ = evl.calculate_auc_roc(Y, Y_)
                if df_mean.empty: df_mean = pd.DataFrame(columns=['mean'])
                df_mean.loc['aucroc'] = aucroc

            if (m := [m for m in metrics.other if 'skill_coverage' in m]):
                log.info(f'{m} ...')
                assert 'skillcoverage' in self.teamsvecs, f'{opentf.textcolor["red"]}Skill coverage metrix is missing! Either remove this metric, or add the matrix to teamsvecs. See https://github.com/fani-lab/OpeNTF/blob/main/src/cmn/team.py#L302{opentf.textcolor["reset"]}'
                X = self.teamsvecs['skill'][self.splits['test']]
                df_skc, df_mean_skc = evl.calculate_skill_coverage(X, Y_, self.teamsvecs['skillcoverage'], per_instance, topks=m[0].replace('skill_coverage_', ''))
                if df.empty: df = df_skc
                else: df = pd.concat([df.reset_index(drop=True), df_skc.reset_index(drop=True)], axis=1)
                if df_mean.empty: df_mean = df_mean_skc
                else: df_mean = pd.concat([df_mean, df_mean_skc], axis=0)
            return df, df_mean

        Y = self.teamsvecs['member'][self.splits['test']]
        log.info(f'{opentf.textcolor["magenta"]}Utility evaluation for {fpred_} ... {opentf.textcolor["reset"]}')
        try:
            log.info(f'Before: Loading {fpred}.eval.mean.csv ...')
            df_before_mean = pd.read_csv(f'{fpred}.eval.mean.csv', names=['mean'], header=0)#we should already have it at f*.test.pred.eval.mean.csv
            if per_instance: df_before = pd.read_csv(f'{fpred}.eval.instance.csv', header=0)
        except FileNotFoundError:
            log.info(f'Before: Loading {fpred}.eval.mean.csv failed! Evaluating from scratch ...')
            df_before, df_before_mean = _evaluate(preds, metrics, per_instance, k_max)
            if per_instance: df_before.to_csv(f'{fpred}.eval.instance.csv', float_format='%.5f', index=False)
            log.info(f'Before: Saving {fpred}.eval.mean.csv ...')
            df_before_mean.to_csv(f'{fpred}.eval.mean.csv')

        if per_instance: df_before.rename(columns={c: f'{c}.before' for c in df_before.columns}, inplace=True)
        df_before_mean.rename(columns={'mean': 'mean.before'}, inplace=True)

        log.info(f'After: Evaluating {fpred_} ...')
        df_after, df_after_mean = _evaluate(preds_, metrics, per_instance, k_max)
        if per_instance: df_after.rename(columns={c: f'{c}.after' for c in df_after.columns}, inplace=True)
        df_after_mean.rename(columns={'mean': 'mean.after'}, inplace=True)
        if per_instance: pd.concat([df_before.reset_index(drop=True), df_after.reset_index(drop=True)], axis=1).to_csv(f'{fpred_}.eval.utility.instance.csv', float_format='%.5f', index=False)
        pd.concat([df_before_mean, df_after_mean], axis=1).to_csv(f'{fpred_}.eval.utility.mean.csv', index_label='metric')
        log.info(f'After: Saved at {fpred_}.eval.utility.mean.csv.')
