from src.airtwinnet.components.data_ingestion import DataIngestion
from src.airtwinnet.components.data_preprocessing import DataPreprocessing
from src.airtwinnet.components.spatio_temporal_alignment import SpatioTemporalAlignment
from src.airtwinnet.components.graph_construction import DynamicUrbanAirQualityGraph
from src.airtwinnet.components.st_gnn import STGNN
from src.airtwinnet.components.prediction_engine import MultiHybridPredictionEngine


def test_end_to_end_components_exist():

    # 1. Data Ingestion
    ingestion = DataIngestion(
        "data/raw/air_quality_sample.csv"
    )

    data = ingestion.initiate_data_ingestion()

    assert ingestion is not None
    assert data is not None


    # 2. Data Preprocessing
    preprocessing = DataPreprocessing(data)

    processed_data = preprocessing.initiate_data_preprocessing()

    assert preprocessing is not None
    assert processed_data is not None


    # 3. Spatio-Temporal Alignment
    alignment = SpatioTemporalAlignment(processed_data)

    aligned_data = alignment.initiate_alignment()

    assert alignment is not None
    assert aligned_data is not None


    # 4. Dynamic Graph Construction
    graph_builder = DynamicUrbanAirQualityGraph

    assert graph_builder is not None


    # 5. ST-GNN
    st_gnn = STGNN(
        input_features=7,
        hidden_features=16
    )

    assert st_gnn is not None


    # 6. Multi-Hybrid Prediction Engine
    prediction_engine = MultiHybridPredictionEngine(
        st_gnn_features=16,
        output_features=3
    )

    assert prediction_engine is not None


    print(
        "End-to-end component integration "
        "structure verified successfully."
    )