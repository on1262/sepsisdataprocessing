import shap
import matplotlib.pyplot as plt

def cal_feature_importance(model, valid_X, fea_names, plot_path, model_type=['gbdt', 'lstm']):
    if model_type == 'gbdt':
        explainer = shap.Explainer(model)
        shap_values = explainer(valid_X)
        shap_values.feature_names = fea_names
        # visualize the first prediction's explanation
        plt.subplots_adjust(left=0.3)
        shap.plots.beeswarm(shap_values, order=shap_values.abs.mean(0), show=False, plot_size=(14,10))
        plt.savefig(plot_path)
        plt.close()
    else:
        assert(0)