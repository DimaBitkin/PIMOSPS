from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import tkinter as tk
import re
from tkinter import messagebox

# -------------------- Модель процессора --------------------
@dataclass
class CPU:
    registers: Dict[str, int] = field(default_factory=lambda: {f"R{i}": 0 for i in range(16)})
    memory: Dict[int, int] = field(default_factory=dict)
    flags: Dict[str, int] = field(default_factory=lambda: {'N': 0, 'Z': 0, 'C': 0, 'V': 0})
    pipeline: List[str] = field(default_factory=lambda: ["" for _ in range(5)])
    pending_write: Optional[Tuple[str, int]] = None
    flush_pipeline: bool = False

    def reset(self):
        self.__init__()

# -------------------- Парсер инструкций --------------------
def parse_instruction(line: str) -> List[str]:
    return line.strip().replace(',', '').split()

# -------------------- Выполнение команд --------------------
def execute(cpu: CPU, instr: List[str]) -> Optional[Tuple[str, int]]:
    if not instr:
        return None

    op = instr[0].upper()

    def get_val(x):
        if x.upper() == 'PC':
            return cpu.registers['R15']
        elif x.upper() == 'LR':
            return cpu.registers['R14']
        elif x.startswith('R'):
            return cpu.registers.get(x.upper(), 0)
        return int(x.strip('#'))

    def check_condition(cond):
        Z, N = cpu.flags['Z'], cpu.flags['N']
        if cond == 'EQ':
            return Z == 1
        elif cond == 'NE':
            return Z == 0
        elif cond == 'GT':
            return Z == 0 and N == 0
        elif cond == 'LT':
            return N == 1
        elif cond == 'GE':
            return N == 0
        elif cond == 'LE':
            return Z == 1 or N == 1
        else:
            return False

    try:
        if op == 'MOV':
            return (instr[1], get_val(instr[2]))

        elif op == 'ADD':
            return (instr[1], get_val(instr[2]) + get_val(instr[3]))

        elif op == 'SUB':
            return (instr[1], get_val(instr[2]) - get_val(instr[3]))

        elif op == 'CMP':
            result = get_val(instr[1]) - get_val(instr[2])
            cpu.flags['Z'] = int(result == 0)
            cpu.flags['N'] = int(result < 0)
            return None

        elif op == 'LDR':
            addr = get_val(instr[2])
            return (instr[1], cpu.memory.get(addr, 0))

        elif op == 'STR':
            addr = get_val(instr[2])
            cpu.memory[addr] = get_val(instr[1])
            return None

        elif op in ['BEQ', 'BNE', 'BGT', 'BLT', 'BGE', 'BLE']:
            cond = op[1:]
            if check_condition(cond):
                cpu.registers['R15'] = int(instr[1])
                cpu.flush_pipeline = True
            return None

        elif op == 'B':
            cpu.registers['R15'] = int(instr[1]) - 1
            cpu.flush_pipeline = True
            return None

        elif op == 'BL':
            cpu.registers['R14'] = cpu.registers['R15'] - 1
            cpu.registers['R15'] = int(instr[1]) - 1
            cpu.flush_pipeline = True
            return None

        elif op == 'BX':
            if instr[1].upper() == 'LR':
                cpu.registers['R15'] = cpu.registers['R14']-1
            else:
                cpu.registers['R15'] = get_val(instr[1])
            cpu.flush_pipeline = True
            return None

        elif op == 'NOP':
            return None

        else:
            raise ValueError(f"Unknown instruction: {op}")

    except IndexError:
        raise ValueError(f"Недостаточно аргументов в инструкции: {' '.join(instr)}")
    except Exception as e:
        raise ValueError(f"Ошибка при выполнении инструкции {' '.join(instr)}: {str(e)}")

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
        keywords = {'MOV', 'ADD', 'SUB', 'CMP', 'B', 'BL', 'BX', 'LDR', 'STR', 'NOP','BEQ', 'BNE', 'BGT', 'BLT', 'BGE', 'BLE'}
        reg_pattern = r'\bR([0-9]|1[0-5])\b'

        for lineno, line in enumerate(lines):
            line_start = f"{lineno+1}.0"
            line_end = f"{lineno+1}.end"
            tokens = parse_instruction(line)
            if tokens and tokens[0].upper() not in keywords:
                self.instr_input.tag_add("error", line_start, line_end)

            for kw in keywords:
                for match in re.finditer(r'\b' + kw + r'\b', line, re.IGNORECASE):
                    start = f"{lineno+1}.{match.start()}"
                    end = f"{lineno+1}.{match.end()}"
                    self.instr_input.tag_add("keyword", start, end)

            for match in re.finditer(reg_pattern, line, re.IGNORECASE):
                start = f"{lineno+1}.{match.start()}"
                end = f"{lineno+1}.{match.end()}"
                self.instr_input.tag_add("register", start, end)

    def load_instructions(self):
        lines = self.instr_input.get("1.0", tk.END).strip().split('\n')
        self.instructions = [parse_instruction(line) for line in lines if line.strip()]
        self.cpu.registers['R15'] = 0
        self.cpu.pipeline = ["" for _ in range(5)]
        self.cpu.pending_write = None
        self.cpu.flush_pipeline = False

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
        if self.cpu.flush_pipeline:
            self.cpu.pipeline = ["" for _ in range(5)]
            self.cpu.flush_pipeline = False

        self.cpu.pipeline.pop()

        if self.cpu.registers['R15'] < len(self.instructions):
            instr_str = " ".join(self.instructions[self.cpu.registers['R15']])

        else:
            instr_str = ""

        self.cpu.pipeline.insert(0, instr_str)
        # Увеличиваем PC только если не было перехода
        if (self.cpu.flush_pipeline == False):
            self.cpu.registers['R15'] += 1

        # Execute stage (index 2)
        if self.cpu.pipeline[2]:
            try:
                result = execute(self.cpu, parse_instruction(self.cpu.pipeline[2]))
                if result:
                    self.cpu.pending_write = result
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return

        # WriteBack stage (index 4)
        if self.cpu.pending_write:
            reg, value = self.cpu.pending_write
            self.cpu.registers[reg] = value
            self.cpu.pending_write = None

        self.update_display()

    def reset(self):
        self.cpu.reset()
        self.instructions = []
        for entry in self.reg_entries.values():
            entry.delete(0, tk.END)
        self.update_display()

    def update_display(self):
        self.state_display.delete("1.0", tk.END)
        self.state_display.insert(tk.END, "== Регистры ==\n")
        for i in range(16):
            reg = f"R{i}"
            self.state_display.insert(tk.END, f"{reg}: {self.cpu.registers[reg]}\n")
        # self.state_display.insert(tk.END, f"PC: {self.cpu.registers['R15']}\n")
        # self.state_display.insert(tk.END, f"LR: {self.cpu.registers['R14']}\n")
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