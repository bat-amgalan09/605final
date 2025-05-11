def train_with_numba(limit=3000, batch_size=64, epochs=10, save_dir='checkpoints/numba_gpu'):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader, vocab_size, tokenizer = prepare_data(batch_size=batch_size, limit=limit)
    model = ChatbotModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses, times, mem_usage, energies, grad_times, accuracies = [], [], [], [], [], [], []
    best_test_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        grad_start = time.time()

        total_loss, total_tokens = 0, 0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, labels)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            tokens = (labels != tokenizer.pad_token_id).sum().item()
            (loss / tokens).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += tokens

        grad_times.append(time.time() - grad_start)
        epoch_time = time.time() - start_time
        times.append(epoch_time)

        mem = torch.cuda.memory_allocated(device) / 1e6
        mem_usage.append(mem)

        cpu_percent = os.getloadavg()[0]  # rough CPU usage proxy
        energies.append(cpu_percent * epoch_time)

        throughput = len(train_loader.dataset) / epoch_time
        avg_train_loss = total_loss / total_tokens
        train_losses.append(avg_train_loss)
        throughputs = [throughput]

        # Evaluation with real accuracy
        model.eval()
        test_loss, test_tokens = 0, 0
        total_correct = 0
        total_label_tokens = 0
        pad_token_id = tokenizer.pad_token_id

        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids, labels)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                test_loss += loss.item()
                test_tokens += (labels != pad_token_id).sum().item()

                pred = logits.argmax(dim=-1)
                correct = ((pred == labels) & (labels != pad_token_id)).sum().item()
                total = (labels != pad_token_id).sum().item()
                total_correct += correct
                total_label_tokens += total

        avg_test_loss = test_loss / test_tokens
        test_losses.append(avg_test_loss)
        accuracy = total_correct / total_label_tokens if total_label_tokens > 0 else 0.0
        accuracies.append(accuracy)

        print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, "
              f"Accuracy = {accuracy:.4f}, Time = {epoch_time:.2f}s, Energy = {energies[-1]:.2f}, Mem = {mem:.2f} MB")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

    return train_losses, test_losses, times, mem_usage, energies, grad_times, accuracies
