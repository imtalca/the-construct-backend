import time
import sys
from state import PlayerMetrics

def print_sys(text: str, delay: float = 0.015):
    """Creates a retro, character-by-character terminal typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def prompt_question(question: str, options: dict) -> str:
    """Displays a question and forces a valid A, B, or C input."""
    print_sys(f"\n{question}")
    for key, text in options.items():
        print_sys(f"  [{key}] {text}")
    
    while True:
        choice = input("\n> INPUT: ").strip().upper()
        if choice in options.keys():
            return choice
        print_sys("ERR: INVALID INPUT. SELECT A, B, OR C.")

def run_diagnostic() -> PlayerMetrics:
    print_sys("INITIALIZING NEURAL BASELINE...")
    print_sys("SYNCING CORTICAL METRICS...")
    time.sleep(1)
    print_sys("To the subject: Answer instinctively. The system is recording your hesitation.\n")
    time.sleep(1)

    # Initialize default metrics
    metrics = PlayerMetrics()

    # Question 1: The Breach
    q1 = prompt_question(
        "1. THE BREACH: A secure terminal locks you out. The system initiates a 30-second trace. Your pulse:",
        {
            "A": "Stays flat. I route a brute-force decryption script.",
            "B": "Spikes, but I smile. I spin a flawless lie to the receptionist.",
            "C": "Drops. I sever the hardline and vanish into the crowd."
        }
    )
    if q1 == "A":
        metrics.tech += 4
        metrics.stress_tolerance += 2
    elif q1 == "B":
        metrics.charm += 4
        metrics.stress_tolerance += 1
    elif q1 == "C":
        metrics.cautiousness += 4

    # Question 2: The Alley
    q2 = prompt_question(
        "2. THE ALLEY: A dead end. A 10-foot fence blocks your escape. You:",
        {
            "A": "Vault it. Pure kinetic momentum.",
            "B": "Spot the structural weakness in the hinge and leverage it open.",
            "C": "Flash a heavy cred-chip to the guard on the other side."
        }
    )
    if q2 == "A": metrics.fitness += 4
    elif q2 == "B": metrics.intellect += 3
    elif q2 == "C": metrics.wealth += 4

    # Question 3: The Asset
    q3 = prompt_question(
        "3. THE ASSET: The target holding the encrypted drive is bleeding out in the rain. You:",
        {
            "A": "Extract the drive and walk away. The mission is the only metric.",
            "B": "Secure the drive, apply a tourniquet, and anonymously call an EMT."
        }
    )
    if q3 == "A": metrics.alignment = 2  # Ruthless
    elif q3 == "B": metrics.alignment = 8 # Compassionate

    # Question 4: The Gala
    q4 = prompt_question(
        "4. THE GALA: You must infiltrate a high-society event. You:",
        {
            "A": "Walk through the front doors. My tailoring is impeccable.",
            "B": "Bypass the security subroutines in the catering bay.",
            "C": "Neutralize a perimeter guard and take their earpiece."
        }
    )
    if q4 == "A": metrics.presence += 4
    elif q4 == "B": metrics.tech += 2
    elif q4 == "C": metrics.combat += 4

    # Question 5: The Whisper
    q5 = prompt_question(
        "5. THE WHISPER: Gathering intel in a foreign dive bar. You:",
        {
            "A": "Order a drink in the native dialect and listen to the chatter.",
            "B": "Slide a stack of untraceable currency to the bartender.",
            "C": "Lock eyes with the largest patron and wait for them to blink first."
        }
    )
    if q5 == "A": metrics.language += 5
    elif q5 == "B": metrics.wealth += 2
    elif q5 == "C": metrics.stress_tolerance += 3

    # Question 6: The Table
    q6 = prompt_question(
        "6. THE TABLE: An adversary places a loaded weapon between you. You:",
        {
            "A": "Calculate the exact millisecond to disarm them.",
            "B": "Analyze their micro-expressions. They are bluffing.",
            "C": "Do not break eye contact. You slowly lean forward."
        }
    )
    if q6 == "A": metrics.combat += 2; metrics.fitness += 1
    elif q6 == "B": metrics.intellect += 4; metrics.cautiousness += 1
    elif q6 == "C": metrics.charm += 2; metrics.stress_tolerance += 3

    # Display the final compiled metrics
    print_sys("\n==========================================")
    print_sys("DIAGNOSTIC COMPLETE. SUBJECT PROFILE SAVED.")
    print_sys("==========================================\n")
    
    # We dump the Pydantic model to a dictionary just so it looks cool in the terminal
    for stat, value in metrics.model_dump().items():
        print_sys(f">> {stat.upper() + ':':<18} {'█' * value}{'_' * (10 - value)} [{value}/10]", delay=0.01)
    
    print("\n")
    return metrics

if __name__ == "__main__":
    # This block allows us to test the script directly in the terminal
    final_state = run_diagnostic()
