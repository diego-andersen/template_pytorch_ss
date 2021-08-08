"""Main control module for model training. Contains the primary training loop."""
import datasets
import trainers
import utils.utils as utils
from utils.iter_tracker import IterationTracker
from options.train_options import TrainOptions

if __name__ == "__main__":
    # Read options
    opt = TrainOptions().parse()

    # Create loader for dataset
    dataloader = datasets.create_dataloader(opt)

    # Create trainer for model
    trainer = trainers.create_trainer(opt)

    # Create training iteration tracker
    iter_tracker = IterationTracker(opt, len(dataloader))

    ### MAIN TRAINING LOOP ###

    for epoch in iter_tracker.training_epochs():
        iter_tracker.record_epoch_start(epoch)

        for data in dataloader:
            # Step through model
            trainer.run_model_one_step(data)
            iter_tracker.record_one_iteration()

            # Check for termination BEFORE writing to disk to avoid corruption
            if utils.instance_is_terminating():
                trainer.terminate_training()

            # Visualizations
            if iter_tracker.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_tracker.epoch_step,
                                                losses, iter_tracker.time_per_iter)
                visualizer.plot_current_errors(losses, iter_tracker.global_step)

            # Save to disk - set opt.save_latest_freq = 1 if you're paranoid
            if iter_tracker.needs_saving():
                print('Saving the latest model (epoch: {:d}, total_steps: {:d})'.format(
                    (epoch, iter_tracker.global_step)))
                trainer.save('latest')
                iter_tracker.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_tracker.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_tracker.total_epochs:
            print("Saving the model at the end of epoch {:d}. Total steps: {:d}".format(
                epoch, iter_tracker.global_step))
            trainer.save('latest')
            trainer.save(epoch)

    print("Training completed successfully")