use debot_db::TransactionLog;
use debot_db::{ModelParams, SerializableModel};
use debot_ml::RandomForest;
use rand::seq::SliceRandom;
use rust_decimal::prelude::ToPrimitive;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::env;
use tokio;

#[tokio::main]
async fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        log::error!("Usage: {} <train|predict> <key> [options]", args[0]);
        return;
    }

    let command = &args[1];
    let key = &args[2];

    let mongodb_uri = env::var("MONGODB_URI").expect("MONGODB_URI must be set");

    let db_name = env::var("DB_NAME").expect("DB_NAME must be set");
    let transaction_log = TransactionLog::new(0, 0, 0, &mongodb_uri, &db_name).await;

    let db_name = env::var("DB_SIM_NAME").expect("DB_SIM_NAME must be set");
    let model_params = ModelParams::new(&mongodb_uri, &db_name).await;

    match command.as_str() {
        "train" => {
            let (x, y) = download_data(&transaction_log, key).await;
            grid_search_and_train(key, &model_params, x, y, 5).await;
        }
        "predict" => {
            let x = generate_single_data_point(key);
            let random_forest = RandomForest::new(key, &model_params).await;
            random_forest.predict(x).await;
        }
        _ => log::error!("Unknown command: {}", command),
    }
}

async fn grid_search_and_train(
    key: &str,
    model_params: &ModelParams,
    x: DenseMatrix<f64>,
    y: Vec<i32>,
    k: usize,
) {
    let initial_param_grid = vec![
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 100,
            m: Some(3),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Some(10),
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 200,
            m: Some(4),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Some(20),
            min_samples_leaf: 2,
            min_samples_split: 5,
            n_trees: 150,
            m: Some(5),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Entropy,
            max_depth: None,
            min_samples_leaf: 2,
            min_samples_split: 3,
            n_trees: 100,
            m: Some(3),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Entropy,
            max_depth: Some(15),
            min_samples_leaf: 1,
            min_samples_split: 4,
            n_trees: 200,
            m: Some(6),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Some(10),
            min_samples_leaf: 2,
            min_samples_split: 2,
            n_trees: 250,
            m: Some(4),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Some(20),
            min_samples_leaf: 1,
            min_samples_split: 5,
            n_trees: 300,
            m: Some(5),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Entropy,
            max_depth: Some(25),
            min_samples_leaf: 2,
            min_samples_split: 3,
            n_trees: 150,
            m: Some(6),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Entropy,
            max_depth: Some(15),
            min_samples_leaf: 1,
            min_samples_split: 4,
            n_trees: 200,
            m: Some(3),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Entropy,
            max_depth: Some(20),
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 250,
            m: Some(5),
            keep_samples: false,
            seed: 42,
        },
    ];

    let mut best_params = &initial_param_grid[0];
    let mut best_accuracy = 0.0;

    for params in &initial_param_grid {
        let avg_accuracy = cross_validate(&x, &y, k, params);
        log::info!(
            "Initial Parameters: {:?}, Average cross-validation accuracy: {}",
            params,
            avg_accuracy
        );
        if avg_accuracy > best_accuracy {
            best_accuracy = avg_accuracy;
            best_params = params;
        }
    }

    log::info!(
        "Best initial parameters: {:?}, Best initial accuracy: {}",
        best_params,
        best_accuracy
    );

    // さらに詳細なパラメータ調整
    let detailed_param_grid = vec![
        RandomForestClassifierParameters {
            criterion: best_params.criterion.clone(),
            max_depth: best_params.max_depth,
            min_samples_leaf: best_params.min_samples_leaf,
            min_samples_split: best_params.min_samples_split,
            n_trees: best_params.n_trees + 50,
            m: best_params.m,
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: best_params.criterion.clone(),
            max_depth: best_params.max_depth,
            min_samples_leaf: best_params.min_samples_leaf,
            min_samples_split: best_params.min_samples_split + 1,
            n_trees: best_params.n_trees + 50,
            m: best_params.m,
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: best_params.criterion.clone(),
            max_depth: best_params.max_depth.map(|d| d + 5),
            min_samples_leaf: best_params.min_samples_leaf + 1,
            min_samples_split: best_params.min_samples_split,
            n_trees: best_params.n_trees + 50,
            m: best_params.m.map(|m| m + 1),
            keep_samples: false,
            seed: 42,
        },
        RandomForestClassifierParameters {
            criterion: best_params.criterion.clone(),
            max_depth: best_params.max_depth.map(|d| d + 5),
            min_samples_leaf: best_params.min_samples_leaf,
            min_samples_split: best_params.min_samples_split,
            n_trees: best_params.n_trees + 100,
            m: best_params.m.map(|m| m + 1),
            keep_samples: false,
            seed: 42,
        },
    ];

    for params in &detailed_param_grid {
        let avg_accuracy = cross_validate(&x, &y, k, params);
        log::info!(
            "Detailed Parameters: {:?}, Average cross-validation accuracy: {}",
            params,
            avg_accuracy
        );
        if avg_accuracy > best_accuracy {
            best_accuracy = avg_accuracy;
            best_params = params;
        }
    }

    log::info!(
        "Best detailed parameters: {:?}, Best detailed accuracy: {}",
        best_params,
        best_accuracy
    );

    // 最適なパラメータでモデルをトレーニングし保存
    let classifier = RandomForestClassifier::fit(&x, &y, best_params.clone()).unwrap();
    let serialized_model = bincode::serialize(&classifier).unwrap();
    let serializable_model = SerializableModel {
        model: serialized_model,
    };
    model_params
        .save_model(key, &serializable_model)
        .await
        .expect("Failed to save model");
}

fn cross_validate(
    x: &DenseMatrix<f64>,
    y: &Vec<i32>,
    k: usize,
    params: &RandomForestClassifierParameters,
) -> f64 {
    let mut indices: Vec<usize> = (0..x.shape().0).collect();
    indices.shuffle(&mut rand::thread_rng());
    let fold_size = x.shape().0 / k;

    let mut accuracies = Vec::new();

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 {
            x.shape().0
        } else {
            (i + 1) * fold_size
        };

        let valid_indices = &indices[start..end];
        let train_indices: Vec<usize> = indices
            .iter()
            .filter(|&&idx| !valid_indices.contains(&idx))
            .copied()
            .collect();

        let x_train = DenseMatrix::from_2d_vec(
            &train_indices
                .iter()
                .map(|&i| x.get_row(i).iterator(0).copied().collect())
                .collect::<Vec<Vec<f64>>>(),
        );
        let y_train: Vec<i32> = train_indices.iter().map(|&i| y[i]).collect();

        let x_valid = DenseMatrix::from_2d_vec(
            &valid_indices
                .iter()
                .map(|&i| x.get_row(i).iterator(0).copied().collect())
                .collect::<Vec<Vec<f64>>>(),
        );
        let y_valid: Vec<i32> = valid_indices.iter().map(|&i| y[i]).collect();

        let classifier = RandomForestClassifier::fit(&x_train, &y_train, params.clone()).unwrap();
        let y_pred = classifier.predict(&x_valid).unwrap();
        let acc = accuracy(&y_valid, &y_pred);
        accuracies.push(acc);
    }

    accuracies.iter().sum::<f64>() / k as f64
}

async fn download_data(
    transaction_log: &TransactionLog,
    key: &str,
) -> (DenseMatrix<f64>, Vec<i32>) {
    let parts: Vec<&str> = key.split('_').collect();
    if parts.len() != 2 {
        panic!("Invalid key format. Expected format: <token_name>_<position_type>");
    }
    let token_name = parts[0];
    let position_type = parts[1];

    // Download price data
    let db = transaction_log.get_db().await.expect("db is none");
    let positions = TransactionLog::get_all_open_positions(&db).await;

    // Collect inputs and outputs from positions
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut outputs: Vec<i32> = Vec::new();

    for position in positions {
        if position.token_name == token_name
            && position.position_type == position_type
            && matches!(
                position.state.as_str(),
                "Closed(TakeProfit)" | "Closed(CutLoss)"
            )
        {
            let debug_log = &position.debug;
            inputs.push(vec![
                debug_log.input_1.to_f64().expect("conversion failed"),
                debug_log.input_2.to_f64().expect("conversion failed"),
                debug_log.input_3.to_f64().expect("conversion failed"),
                debug_log.input_4.to_f64().expect("conversion failed"),
                debug_log.input_5.to_f64().expect("conversion failed"),
                debug_log.input_6.to_f64().expect("conversion failed"),
                debug_log.input_7.to_f64().expect("conversion failed"),
                debug_log.input_8.to_f64().expect("conversion failed"),
                debug_log.input_9.to_f64().expect("conversion failed"),
                debug_log.input_10.to_f64().expect("conversion failed"),
            ]);
            outputs.push(debug_log.output_1.to_i32().expect("conversion failed"));
        }
    }

    log::info!("num of inputs = {}", inputs.len());

    let input_slices: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();
    let x = DenseMatrix::from_2d_array(&input_slices);
    (x, outputs)
}

fn generate_single_data_point(position_type: &str) -> DenseMatrix<f64> {
    if position_type == "Long" {
        DenseMatrix::from_2d_array(&[&[
            0.3251, 89.0097, 0.1404, -25.1563, 0.1, 0.001, 1711.5156, 0.1494, 0.0, 0.0,
        ]])
    } else {
        DenseMatrix::from_2d_array(&[&[
            0.4376, 105.4768, 0.1439, -8.5096, 0.1, 0.002, 279.1822, 0.1006, 0.0, 0.0,
        ]])
    }
}
