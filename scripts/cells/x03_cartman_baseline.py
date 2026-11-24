bl = json.load(open('outputs/cartman_baseline.json'))
names = {'gt_pairwise': 'GT: 3 референса попарно', 'cartman': 'другие фразы Картмана', 'kids': 'другие дети (MTV)',
         'adults': 'взрослые персонажи (MTV)', 'smaug': 'Смауг'}
pd.DataFrame({names[k]: {'similarity': v['mean'], 'std': v['std'], 'доля пар > 0.86': v.get('share_gt_086')}
              for k, v in bl.items()}).T.round(3)
