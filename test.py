def test(net, test_loader, criterion, optimizer):
    # To keep all the losses
    test_losses = []
    # To keep a count of the correct predictions
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size)

    # Since we are evaluating
    net.eval()

    # Start the loop
    for inputs, labels in test_loader:
        # Create new variables for the hidden state
        h = tuple([each.data for each in h])

        # Move to GPU
        if (net.train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Get predicted outputs
        output, h = net(inputs, h)

        # Calculate the loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss)

        # convert output prob to predicted class
        pred = torch.round(output.squeeze())

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        if not self.train_on_gpu:
            correct = np.squeeze(correct_tensor.numpy())
        else:
            correct = np.squeeze(correct_tensor.cpu().numpy())

        num_correct += np.sum(correct)

    ### Stats ###
    print ("Test Loss: {:3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print ('Test accuracy: {:3f}'.format(test_acc))


def predict(net, test_review, seq_length=200):

    test_review = test_review.lower()
    tr = ''.join(c for c in test_review if c not in punctuation)
    tr = tr.split()

    tints = []
    tints.append([vocab_to_int[word] for word in tr])

    f = pad_features(tints, seq_length)

    net.eval()

    ftensor = torch.from_numpy(f)

    batch_size = ftensor.shape[0]

    h = net.init_hidden(batch_size)

    if (torch.cuda.is_available()):
        net = net.cuda()
        ftensor = ftensor.cuda()

    out, h = net(ftensor, h)

    pred = torch.round(out.squeeze())

    print ('Probabilities before rounding: {:6f}'.format(out.item()))

    if pred.item() == 1:
        print ('The sentiment is positive')
    else:
        print ('The sentiment is negetive')
