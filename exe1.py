# arm_simulator.py

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional
import tkinter as tk
import re
from tkinter import messagebox

# -------------------- Модель процессора --------------------
@dataclass
class CPU:
    registers: Dict[str, int] = field(default_factory=lambda: {f"R{i}": 0 for i in range(16)})
    memory: Dict[int, int] = field(default_factory=dict)
    pc: int = 0
    lr: int = 0  # Link Register (используется для BL)
    flags: Dict[str, int] = field(default_factory=lambda: {'N': 0, 'Z': 0, 'C': 0, 'V': 0})
    pipeline: List[str] = field(default_factory=lambda: ["" for _ in range(5)])  # F, D, E, M, W

    def reset(self):
        self.__init__()

# -------------------- Парсер инструкций --------------------
def parse_instruction(line: str) -> List[str]:
    return line.strip().replace(',', '').split()

# -------------------- Выполнение команд --------------------
def execute(cpu: CPU, instr: List[str]):
    if not instr:
        return

    op = instr[0].upper()

    def get_val(x):
        if x.startswith('R'):
            return cpu.registers.get(x, 0)
        return int(x.strip('#'))

    def get_imm(x):
        return int(x.strip('#'))

    if op == 'MOV':
        cpu.registers[instr[1]] = get_val(instr[2])

    elif op == 'ADD':
        cpu.registers[instr[1]] = get_val(instr[2]) + get_val(instr[3])

    elif op == 'SUB':
        cpu.registers[instr[1]] = get_val(instr[2]) - get_val(instr[3])

    elif op == 'CMP':
        result = get_val(instr[1]) - get_val(instr[2])
        cpu.flags['Z'] = int(result == 0)
        cpu.flags['N'] = int(result < 0)

    elif op == 'LDR':
        addr = get_val(instr[2])
        cpu.registers[instr[1]] = cpu.memory.get(addr, 0)

    elif op == 'STR':
        addr = get_val(instr[2])
        cpu.memory[addr] = get_val(instr[1])

    elif op == 'B':
        cpu.pc = int(instr[1]) - 1

    elif op == 'BL':
        cpu.lr = cpu.pc
        cpu.pc = int(instr[1]) - 1

    elif op == 'BX':
        if instr[1].upper() == 'LR':
            cpu.pc = cpu.lr
        else:
            cpu.pc = get_val(instr[1])

    elif op == 'NOP':
        pass

    else:
        raise ValueError(f"Unknown instruction: {op}")

# -------------------- UI-интерфейс --------------------
class SimulatorApp:
    def __init__(self, root):
        self.root = root
        self.cpu = CPU()
        self.instructions: List[List[str]] = []

        root.title("ARM Instruction Simulator")

        self.instr_input = tk.Text(root, height=10, width=50)
        self.instr_input.pack()
        self.instr_input.bind("<KeyRelease>", self.syntax_highlight)

        # Определяем стили для подсветки
        self.instr_input.tag_configure("keyword", foreground="blue")
        self.instr_input.tag_configure("register", foreground="dark green")
        self.instr_input.tag_configure("error", background="red", foreground="white")

        self.reg_frame = tk.LabelFrame(root, text="Начальные значения регистров")
        self.reg_frame.pack()
        self.reg_entries = {}
        for i in range(16):
            reg = f"R{i}"
            lbl = tk.Label(self.reg_frame, text=reg)
            lbl.grid(row=i // 8, column=(i % 8) * 2)
            ent = tk.Entry(self.reg_frame, width=5)
            ent.grid(row=i // 8, column=(i % 8) * 2 + 1)
            self.reg_entries[reg] = ent

        self.load_button = tk.Button(root, text="Загрузить", command=self.load_instructions)
        self.load_button.pack()

        self.step_button = tk.Button(root, text="Следующий такт", command=self.step)
        self.step_button.pack()

        self.reset_button = tk.Button(root, text="Сброс", command=self.reset)
        self.reset_button.pack()

        self.state_display = tk.Text(root, height=30, width=70)
        self.state_display.pack()
        self.update_display()

    def syntax_highlight(self, event=None):
        text = self.instr_input.get("1.0", tk.END)
        self.instr_input.tag_remove("keyword", "1.0", tk.END)
        self.instr_input.tag_remove("register", "1.0", tk.END)
        self.instr_input.tag_remove("error", "1.0", tk.END)

        lines = text.strip().split("\n")
        keywords = {'MOV', 'ADD', 'SUB', 'CMP', 'B', 'BL', 'BX', 'LDR', 'STR', 'NOP'}
        reg_pattern = r'\bR([0-9]|1[0-5])\b'
        
        for lineno, line in enumerate(lines):
            line_start = f"{lineno+1}.0"
            line_end = f"{lineno+1}.end"

            # Проверка синтаксиса
            tokens = parse_instruction(line)
            if tokens and tokens[0].upper() not in keywords:
                self.instr_input.tag_add("error", line_start, line_end)

            # Подсветка ключевых слов
            for kw in keywords:
                for match in re.finditer(r'\b' + kw + r'\b', line, re.IGNORECASE):
                    start = f"{lineno+1}.{match.start()}"
                    end = f"{lineno+1}.{match.end()}"
                    self.instr_input.tag_add("keyword", start, end)

            # Подсветка регистров
            for match in re.finditer(reg_pattern, line, re.IGNORECASE):
                start = f"{lineno+1}.{match.start()}"
                end = f"{lineno+1}.{match.end()}"
                self.instr_input.tag_add("register", start, end)

    def load_instructions(self):
        lines = self.instr_input.get("1.0", tk.END).strip().split('\n')
        self.instructions = [parse_instruction(line) for line in lines if line.strip()]
        self.cpu.pc = 0
        self.cpu.pipeline = ["" for _ in range(5)]

        # Применение значений регистров из формы
        for reg, entry in self.reg_entries.items():
            val = entry.get()
            if val:
                try:
                    self.cpu.registers[reg] = int(val)
                except ValueError:
                    messagebox.showerror("Ошибка", f"Неверное значение для {reg}: {val}")
                    return

        self.update_display()

    def step(self):
        # Обработка конвейера
        self.cpu.pipeline.pop()
        instr_str = " ".join(self.instructions[self.cpu.pc]) if self.cpu.pc < len(self.instructions) else ""
        self.cpu.pipeline.insert(0, instr_str)

        # Исполнение только когда команда на стадии Execute
        if self.cpu.pipeline[2]:
            try:
                execute(self.cpu, parse_instruction(self.cpu.pipeline[2]))
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return

        if self.cpu.pc < len(self.instructions):
            self.cpu.pc += 1

        self.update_display()

    def reset(self):
        self.cpu.reset()
        self.instructions = []
        self.instr_input.delete("1.0", tk.END)
        for entry in self.reg_entries.values():
            entry.delete(0, tk.END)
        self.update_display()

    def update_display(self):
        self.state_display.delete("1.0", tk.END)
        self.state_display.insert(tk.END, "== Регистры ==\n")
        for i in range(16):
            reg = f"R{i}"
            self.state_display.insert(tk.END, f"{reg}: {self.cpu.registers[reg]}\n")
        self.state_display.insert(tk.END, f"PC: {self.cpu.pc}\n")
        self.state_display.insert(tk.END, f"LR: {self.cpu.lr}\n")
        self.state_display.insert(tk.END, "\n== Флаги ==\n")
        for flag, val in self.cpu.flags.items():
            self.state_display.insert(tk.END, f"{flag}: {val}  ")

        self.state_display.insert(tk.END, "\n\n== Конвейер ==\n")
        stages = ["Fetch", "Decode", "Execute", "Memory", "WriteBack"]
        for i, instr in enumerate(self.cpu.pipeline):
            self.state_display.insert(tk.END, f"{stages[i]}: {instr}\n")

        self.state_display.insert(tk.END, "\n== Память ==\n")
        for addr in sorted(self.cpu.memory.keys()):
            self.state_display.insert(tk.END, f"[{addr}]: {self.cpu.memory[addr]}\n")

# -------------------- Запуск --------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = SimulatorApp(root)
    root.mainloop()
qwert