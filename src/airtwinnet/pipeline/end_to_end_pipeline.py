class AirTwinNetPipeline:

    def __init__(
        self,
        ingestion,
        preprocessing,
        alignment,
        fusion,
        digital_twin,
        graph_builder,
        st_gnn,
        prediction_engine,
        xai,
        feedback_loop
    ):

        self.ingestion = ingestion
        self.preprocessing = preprocessing
        self.alignment = alignment
        self.fusion = fusion
        self.digital_twin = digital_twin
        self.graph_builder = graph_builder
        self.st_gnn = st_gnn
        self.prediction_engine = prediction_engine
        self.xai = xai
        self.feedback_loop = feedback_loop

    def run(self):

        print("Starting AirTwinNet end-to-end pipeline...")

        data = self.ingestion.initiate_data_ingestion()

        data = self.preprocessing.preprocess(data)

        data = self.alignment.align(data)

        fused_data = self.fusion.fuse(data)

        twin_state = self.digital_twin.update_state(
            fused_data
        )

        graph = self.graph_builder.build_graph(
            twin_state
        )

        representation = self.st_gnn.generate_representation(
            graph
        )

        predictions = self.prediction_engine.predict(
            representation
        )

        print(
            "AirTwinNet end-to-end pipeline "
            "completed successfully."
        )

        return {
            "data": data,
            "twin_state": twin_state,
            "graph": graph,
            "representation": representation,
            "predictions": predictions
        }
    