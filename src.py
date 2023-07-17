import os
import openai
openai.api_key = "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"
import nltk
import re
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from nltk.corpus import words

nltk.download("words")
class SpellChecker:

    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("600x500")

        self.text = ScrolledText(self.root, font=("Ariel", 12))
        self.text.bind("<KeyRelease>", self.check)
        self.text.pack()
        self.old_spaces = 0

        self.root.mainloop()

    def chat_bot(prompt, model = "text-davinci-003"):
        response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=1024, n=1, stop = None, temperature=0.5,)
        return response.choices[0].text



    def check(self, event):
        content = self.text.get("1.0", tk.END)
        space_count = content.count(" ")

        for tag in self.text.tag_names():
            self.text.tag_delete(tag)

        # isolates individual words
        if space_count != self.old_spaces:
            self.old_spaces = space_count
            # removes special characters
            for word in content.split(" "):
                # if word not in word library changes it to the color red
                if re.sub(r"[^\w]", "", word.lower()) not in words.words():
                    position = content.find(word)
                    self.text.tag_add(word, f"1.{position}", f"1.{position + len(word)}")
                    gptInput = "correct the spelling of " + word + " and return nothing but the corrected word";
                    newWord = self.chat_bot(gptInput)
                    #self.text.tag_config(word, foreground="red")
                    self.text.delete(word, f"1.{position}", f"1.{position + len(word)}")
                    self.text.insert(f"1.{position}", newWord)



SpellChecker()
