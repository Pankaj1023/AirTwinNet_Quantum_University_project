class AirTwinNetPipeline:

    def __init__(
        self,
        ingestion,
        preprocessing_class,
        alignment_class,
        fusion_class,
        digital_twin_class,
        graph_builder_class,
        st_gnn,
        prediction_engine,
        xai,
        feedback_loop
    ):

        self.ingestion = ingestion
        self.preprocessing_class = preprocessing_class
        self.alignment_class = alignment_class
        self.fusion_class = fusion_class
        self.digital_twin_class = digital_twin_class
        self.graph_builder_class = graph_builder_class
        self.st_gnn = st_gnn
        self.prediction_engine = prediction_engine
        self.xai = xai
        self.feedback_loop = feedback_loop

    def run(self):

        print("Starting AirTwinNet end-to-end pipeline...")

        # --------------------------------------------------
        # 1. Data Ingestion
        # --------------------------------------------------

        data = self.ingestion.initiate_data_ingestion()

        # --------------------------------------------------
        # 2. Data Preprocessing
        # --------------------------------------------------

        preprocessing = self.preprocessing_class(data)

        data = preprocessing.initiate_data_preprocessing()

        # --------------------------------------------------
        # 3. Spatio-Temporal Alignment
        # --------------------------------------------------

        alignment = self.alignment_class(data)

        data = alignment.initiate_alignment()

        # --------------------------------------------------
        # 4. Multi-Modal Data Fusion
        # --------------------------------------------------

        fusion = self.fusion_class(data)

        fused_data = fusion.initiate_data_fusion()

        # --------------------------------------------------
        # 5. Digital Twin State Update
        # --------------------------------------------------

        digital_twin = self.digital_twin_class(fused_data)

        twin_state = digital_twin.update_state(
            fused_data
        )

        # --------------------------------------------------
        # 6. Dynamic Urban Air Quality Graph
        # --------------------------------------------------

        graph_builder = self.graph_builder_class(
            twin_state
        )

        graph = graph_builder.build_graph()

        # --------------------------------------------------
        # 7. Prepare ST-GNN Node Features
        # --------------------------------------------------

        node_features = []

        for node in graph["nodes"]:

            features = graph["node_features"][node]

            node_features.append(
                list(features.values())
            )

        # --------------------------------------------------
        # 8. ST-GNN Representation
        # --------------------------------------------------

        representation = self.st_gnn.generate_representation(
            node_features
        )

        # --------------------------------------------------
        # 9. Prediction
        # --------------------------------------------------

        predictions = self.prediction_engine.predict(
            representation
        )

        print(
            "AirTwinNet end-to-end pipeline "
            "completed successfully."
        )

        return {
            "data": data,
            "fused_data": fused_data,
            "twin_state": twin_state,
            "graph": graph,
            "representation": representation,
            "predictions": predictions
        }