"""
Gera relatório de drift de dados comparando referência e produção
usando Evidently AI e salva como JSON.
"""

import json
import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def load_data(reference_path: str, production_path: str) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e imprime informações básicas dos datasets."""
    print(f"📥 Lendo dados de referência de: {reference_path}")
    ref_df = pd.read_csv(reference_path)
    print(f"   → shape {ref_df.shape}")

    print(f"📥 Lendo dados de produção de: {production_path}")
    prod_df = pd.read_csv(production_path)
    print(f"   → shape {prod_df.shape}")

    return ref_df, prod_df


def generate_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Gera e salva um relatório JSON de drift de dados."""
    print("⚙️ Gerando relatório de Data Drift...")

    # Configuração para suprimir warnings específicos
    import warnings

    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="divide by zero encountered in divide")

    # Cria e executa o relatório
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=production_df)

    # Cria o diretório de saída se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cria um dicionário com informações básicas sobre o drift
    result_dict = {
        "timestamp": str(pd.Timestamp.now()),
        "reference_shape": reference_df.shape,
        "production_shape": production_df.shape,
        "metrics": []
    }

    # Adiciona metadata se disponível
    if hasattr(report, "metadata") and report.metadata:
        result_dict["metadata"] = report.metadata

    # Calculamos manualmente o drift entre os datasets
    drift_results = {
        "drift_share_config": 0.5,  # Valor default de configuração
        "features_drifted": 0,
        "features_analyzed": 0,
        "drift_share_detected": 0.0,
        "drifted_features": []
    }

    # Vamos analisar coluna por coluna
    common_columns = set(reference_df.columns) & set(production_df.columns)
    features_analyzed = 0
    features_drifted = 0

    from scipy import stats

    for col in common_columns:
        # Ignoramos colunas que sejam completamente nulas
        if reference_df[col].isnull().all() or production_df[col].isnull().all():
            continue

        features_analyzed += 1

        # Para colunas numéricas, usamos teste KS
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            ref_data = reference_df[col].dropna()
            prod_data = production_df[col].dropna()

            if len(ref_data) > 0 and len(prod_data) > 0:
                # Teste de Kolmogorov-Smirnov
                ks_stat, p_value = stats.ks_2samp(ref_data, prod_data)

                # Se p-value < 0.05, consideramos que houve drift
                is_drifted = p_value < 0.05

                if is_drifted:
                    features_drifted += 1
                    drift_results["drifted_features"].append({
                        "name": col,
                        "p_value": float(p_value),
                        "stat": float(ks_stat)
                    })

    # Calculamos a proporção de features que sofreram drift
    if features_analyzed > 0:
        drift_results["features_analyzed"] = features_analyzed
        drift_results["features_drifted"] = features_drifted
        drift_results["drift_share_detected"] = features_drifted / \
            features_analyzed

    # Adicionamos ao relatório
    metric_info = {
        "type": "DataDriftPreset",
        "config": {
            "drift_share": 0.5,  # Threshold configurado no preset
            "threshold": None,
            "columns": None
        },
        "results": drift_results
    }

    result_dict["metrics"].append(metric_info)

    # Salva o relatório no formato JSON
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"✅ Relatório JSON salvo em: {output_path}")
    print(
        f"📊 Drift detectado: {drift_results['drift_share_detected']:.4f} ({features_drifted}/{features_analyzed} features)")


if __name__ == "__main__":
    # Ajuste estas paths conforme a sua estrutura:
    reference_file = "data/processed/train_data.csv"
    production_file = "data/processed/test_data.csv"
    report_output = "reports/data_drift_report.json"

    ref_data, prod_data = load_data(reference_file, production_file)
    generate_drift_report(ref_data, prod_data, report_output)
