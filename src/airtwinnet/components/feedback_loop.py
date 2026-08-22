class AirTwinNetFeedbackLoop:

    def __init__(
        self,
        error_monitor,
        retraining_manager
    ):

        self.error_monitor = error_monitor
        self.retraining_manager = retraining_manager

    def process_feedback(
        self,
        actual,
        predicted
    ):

        performance = (
            self.error_monitor.monitor_performance(
                actual,
                predicted,
                self.retraining_manager.error_threshold
            )
        )

        return performance

    def execute_retraining_if_required(
        self,
        model,
        training_data,
        performance
    ):

        if performance["retraining_required"]:

            updated_model = (
                self.retraining_manager.update_model(
                    model,
                    training_data
                )
            )

            return {
                "retrained": True,
                "model": updated_model
            }

        return {
            "retrained": False,
            "model": model
        }