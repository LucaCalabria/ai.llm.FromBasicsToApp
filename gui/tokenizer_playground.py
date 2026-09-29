"""Small Tkinter workbench for the project's byte-pair tokenizer."""

from __future__ import annotations

import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk

from tokenization.common import get_most_frequent_pair, merge_idx, text_to_idx
from tokenization.tokenizer import BaseTokenizer


class TokenizerPlayground(tk.Tk):
    """Provide a GUI for training, encoding, decoding, and byte utilities."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Tokenizer Playground")
        self.geometry("960x720")
        self.minsize(760, 580)

        self.tokenizer: BaseTokenizer | None = None
        self.utility_bytes: list[int] = []
        self.utility_pair: tuple[int, int] | None = None

        self._configure_style()
        self._build_widgets()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f4")
        style.configure("TLabel", background="#f3f5f4", foreground="#1f302d")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#173e38")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#5c706b")
        style.configure("TButton", padding=(12, 7), font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook", background="#f3f5f4", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe", background="#f3f5f4", bordercolor="#d4dfdc")
        style.configure("TLabelframe.Label", background="#f3f5f4", foreground="#173e38")

    def _build_widgets(self) -> None:
        page = ttk.Frame(self, padding=22)
        page.pack(fill="both", expand=True)

        ttk.Label(page, text="Tokenizer Playground", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            page,
            text="Train byte-pair merges, inspect token IDs, and experiment with byte utilities.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 16))

        self.status_var = tk.StringVar(value="Train a tokenizer to begin.")
        notebook = ttk.Notebook(page)
        notebook.pack(fill="both", expand=True)

        tokenizer_tab = ttk.Frame(notebook, padding=16)
        utility_tab = ttk.Frame(notebook, padding=16)
        notebook.add(tokenizer_tab, text="Tokenizer")
        notebook.add(utility_tab, text="Byte Utilities")

        self._build_tokenizer_tab(tokenizer_tab)
        self._build_utility_tab(utility_tab)

        ttk.Label(page, textvariable=self.status_var, style="Subtitle.TLabel").pack(
            anchor="w", pady=(12, 0)
        )

    def _build_tokenizer_tab(self, parent: ttk.Frame) -> None:
        text_group = ttk.LabelFrame(parent, text="Training text", padding=12)
        text_group.pack(fill="both", expand=True)
        self.input_text = tk.Text(
            text_group,
            height=8,
            wrap="word",
            font=("Consolas", 10),
            background="#ffffff",
            foreground="#1f302d",
            insertbackground="#1f302d",
            relief="flat",
            padx=10,
            pady=9,
        )
        self.input_text.pack(fill="both", expand=True)
        self.input_text.insert("1.0", "ababab tokenization is fun")

        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=12)
        ttk.Label(controls, text="Vocabulary size").pack(side="left")
        self.vocab_size = tk.StringVar(value="270")
        ttk.Spinbox(
            controls,
            from_=256,
            to=65536,
            width=9,
            textvariable=self.vocab_size,
        ).pack(side="left", padx=(8, 18))
        ttk.Button(controls, text="Train", command=self._train).pack(side="left")
        ttk.Button(controls, text="Encode", command=self._encode).pack(side="left", padx=8)
        ttk.Button(controls, text="Decode IDs", command=self._decode).pack(side="left")

        token_group = ttk.LabelFrame(parent, text="Token IDs (comma-separated; editable)", padding=12)
        token_group.pack(fill="x", pady=(0, 12))
        self.token_ids = tk.StringVar()
        ttk.Entry(token_group, textvariable=self.token_ids, font=("Consolas", 10)).pack(
            fill="x", expand=True
        )

        decoded_group = ttk.LabelFrame(parent, text="Decoded text", padding=12)
        decoded_group.pack(fill="both", expand=True)
        self.decoded_text = tk.Text(
            decoded_group,
            height=5,
            wrap="word",
            font=("Consolas", 10),
            background="#ffffff",
            foreground="#1f302d",
            relief="flat",
            padx=10,
            pady=9,
        )
        self.decoded_text.pack(fill="both", expand=True)

    def _build_utility_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Run the common byte helpers on an input string.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        self.utility_input = tk.Text(
            parent,
            height=5,
            wrap="word",
            font=("Consolas", 10),
            background="#ffffff",
            foreground="#1f302d",
            relief="flat",
            padx=10,
            pady=9,
        )
        self.utility_input.pack(fill="x")
        self.utility_input.insert("1.0", "byte pair byte pair")

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Analyze bytes", command=self._analyze_bytes).pack(side="left")
        ttk.Label(actions, text="New token ID").pack(side="left", padx=(18, 8))
        self.new_token_id = tk.StringVar(value="256")
        ttk.Spinbox(
            actions,
            from_=256,
            to=65536,
            width=9,
            textvariable=self.new_token_id,
        ).pack(side="left")
        ttk.Button(actions, text="Merge frequent pair", command=self._merge_utility_pair).pack(
            side="left", padx=12
        )

        result_group = ttk.LabelFrame(parent, text="Utility results", padding=12)
        result_group.pack(fill="both", expand=True)
        self.utility_results = tk.Text(
            result_group,
            wrap="word",
            font=("Consolas", 10),
            background="#ffffff",
            foreground="#1f302d",
            relief="flat",
            padx=10,
            pady=9,
            state="disabled",
        )
        self.utility_results.pack(fill="both", expand=True)

    def _get_input_text(self) -> str:
        return self.input_text.get("1.0", "end-1c")

    def _train(self) -> None:
        try:
            vocab_size = int(self.vocab_size.get())
            if vocab_size < 256:
                raise ValueError("Vocabulary size must be at least 256.")
            text = self._get_input_text()
            if not text:
                raise ValueError("Enter training text before training.")

            self.tokenizer = BaseTokenizer()
            self.tokenizer.train(text, vocab_size)
            self.status_var.set(
                f"Trained {len(self.tokenizer.idx_merges)} merges; vocabulary has "
                f"{len(self.tokenizer.vocab)} entries."
            )
        except (ValueError, TypeError, IndexError) as error:
            messagebox.showerror("Training failed", str(error), parent=self)

    def _encode(self) -> None:
        if self.tokenizer is None:
            messagebox.showinfo("Train first", "Train a tokenizer before encoding text.", parent=self)
            return
        try:
            token_ids = self.tokenizer.encode(self._get_input_text())
            self.token_ids.set(", ".join(str(token_id) for token_id in token_ids))
            self._set_text(self.decoded_text, self.tokenizer.decode(token_ids))
            self.status_var.set(f"Encoded text into {len(token_ids)} tokens.")
        except (KeyError, ValueError, TypeError) as error:
            messagebox.showerror("Encoding failed", str(error), parent=self)

    def _decode(self) -> None:
        if self.tokenizer is None:
            messagebox.showinfo("Train first", "Train a tokenizer before decoding IDs.", parent=self)
            return
        try:
            raw_ids = self.token_ids.get().strip()
            token_ids = [int(value.strip()) for value in raw_ids.split(",") if value.strip()]
            decoded = self.tokenizer.decode(token_ids)
            self._set_text(self.decoded_text, decoded)
            self.status_var.set(f"Decoded {len(token_ids)} token IDs.")
        except (KeyError, ValueError, TypeError) as error:
            messagebox.showerror("Decoding failed", str(error), parent=self)

    def _analyze_bytes(self) -> None:
        text = self.utility_input.get("1.0", "end-1c")
        self.utility_bytes = text_to_idx(text)
        self.utility_pair = get_most_frequent_pair(self.utility_bytes)
        pair_counts = Counter(zip(self.utility_bytes, self.utility_bytes[1:]))
        frequent_pair_text = (
            f"{self.utility_pair} (count {pair_counts[self.utility_pair]})"
            if self.utility_pair is not None
            else "No adjacent byte pair found."
        )
        details = (
            f"UTF-8 bytes ({len(self.utility_bytes)}):\n{self.utility_bytes}\n\n"
            f"Most frequent adjacent pair: {frequent_pair_text}"
        )
        self._set_text(self.utility_results, details)
        self.status_var.set("Byte analysis complete.")

    def _merge_utility_pair(self) -> None:
        if not self.utility_bytes:
            self._analyze_bytes()
        if self.utility_pair is None:
            messagebox.showinfo("No pair", "Enter at least two bytes to merge a pair.", parent=self)
            return
        try:
            new_token = int(self.new_token_id.get())
            merged = merge_idx(self.utility_bytes, self.utility_pair, new_token)
            current = self.utility_results.get("1.0", "end-1c")
            self._set_text(
                self.utility_results,
                f"{current}\n\nMerged with token {new_token}:\n{merged}",
            )
            self.status_var.set(f"Merged occurrences of byte pair {self.utility_pair}.")
        except (ValueError, TypeError) as error:
            messagebox.showerror("Merge failed", str(error), parent=self)

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        was_disabled = widget.cget("state") == "disabled"
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        if was_disabled:
            widget.configure(state="disabled")


if __name__ == "__main__":
    app = TokenizerPlayground()
    app.mainloop()