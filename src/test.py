#!/usr/bin/env python3
"""
Erweiterte Plausibilitätsprüfung für synthetische Daten
Fokus auf logische Konsistenz und Geschäftsregeln
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import json
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    ERROR = "ERROR"      # Schwerwiegender logischer Fehler
    WARNING = "WARNING"  # Unwahrscheinlich aber möglich
    INFO = "INFO"       # Leichte Inkonsistenz

@dataclass
class PlausibilityRule:
    """Definition einer Plausibilitätsregel"""
    name: str
    description: str
    check_function: Callable
    level: ValidationLevel
    category: str

@dataclass
class ValidationResult:
    """Ergebnis einer Validierungsprüfung"""
    rule_name: str
    passed: bool
    level: ValidationLevel
    message: str
    affected_fields: List[str]
    row_index: int = None

class PlausibilityChecker:
    """
    Umfassender Plausibilitätschecker für synthetische Daten
    Fokus auf logische Konsistenz und realistische Datenkombinationen
    """
    
    def __init__(self):
        self.rules = []
        self.results = []
        
    def add_rule(self, rule: PlausibilityRule):
        """Fügt eine neue Plausibilitätsregel hinzu"""
        self.rules.append(rule)
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validiert gesamten Datensatz"""
        self.results = []
        
        print(f"Validiere {len(df)} Datensätze mit {len(self.rules)} Regeln...")
        
        for idx, row in df.iterrows():
            row_results = self._validate_row(row, idx)
            self.results.extend(row_results)
        
        return self._generate_summary()
    
    def _validate_row(self, row: pd.Series, row_idx: int) -> List[ValidationResult]:
        """Validiert einzelne Zeile gegen alle Regeln"""
        row_results = []
        
        for rule in self.rules:
            try:
                result = rule.check_function(row)
                if isinstance(result, ValidationResult):
                    result.row_index = row_idx
                    row_results.append(result)
                elif isinstance(result, bool):
                    # Fallback für einfache Boolean-Rückgaben
                    row_results.append(ValidationResult(
                        rule_name=rule.name,
                        passed=result,
                        level=rule.level,
                        message=f"Rule {rule.name}: {'PASSED' if result else 'FAILED'}",
                        affected_fields=[],
                        row_index=row_idx
                    ))
            except Exception as e:
                # Fehler bei Regelausführung
                row_results.append(ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Fehler bei Regelausführung: {str(e)}",
                    affected_fields=[],
                    row_index=row_idx
                ))
        
        return row_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generiert Zusammenfassung der Validierungsergebnisse"""
        total_checks = len(self.results)
        failed_checks = len([r for r in self.results if not r.passed])
        
        # Gruppierung nach Validierungslevel
        errors = [r for r in self.results if r.level == ValidationLevel.ERROR and not r.passed]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING and not r.passed]
        infos = [r for r in self.results if r.level == ValidationLevel.INFO and not r.passed]
        
        # Häufigste Regelverstöße
        rule_violations = {}
        for result in self.results:
            if not result.passed:
                rule_violations[result.rule_name] = rule_violations.get(result.rule_name, 0) + 1
        
        return {
            'total_checks': total_checks,
            'failed_checks': failed_checks,
            'success_rate': (total_checks - failed_checks) / total_checks if total_checks > 0 else 0,
            'errors': len(errors),
            'warnings': len(warnings),
            'infos': len(infos),
            'rule_violations': rule_violations,
            'detailed_results': self.results
        }
    
    def print_summary(self, summary: Dict[str, Any]):
        """Druckt übersichtliche Zusammenfassung"""
        print("\n" + "="*70)
        print("                PLAUSIBILITÄTSPRÜFUNG - ERGEBNISSE")
        print("="*70)
        
        print(f"\n📊 GESAMTSTATISTIK:")
        print(f"   • Durchgeführte Prüfungen: {summary['total_checks']:,}")
        print(f"   • Fehlgeschlagene Prüfungen: {summary['failed_checks']:,}")
        print(f"   • Erfolgsrate: {summary['success_rate']:.1%}")
        
        print(f"\n🚨 NACH SCHWEREGRAD:")
        print(f"   • Schwere Fehler (ERROR): {summary['errors']}")
        print(f"   • Warnungen (WARNING): {summary['warnings']}")
        print(f"   • Hinweise (INFO): {summary['infos']}")
        
        if summary['rule_violations']:
            print(f"\n🔍 HÄUFIGSTE REGELVERSTÖSSE:")
            sorted_violations = sorted(summary['rule_violations'].items(), 
                                     key=lambda x: x[1], reverse=True)
            for rule_name, count in sorted_violations[:5]:
                print(f"   • {rule_name}: {count} Verstöße")
        
        print("\n" + "="*70)

# Sammlung von Standard-Plausibilitätsregeln
class StandardPlausibilityRules:
    """Sammlung von häufig genutzten Plausibilitätsregeln"""
    
    @staticmethod
    def age_employment_consistency():
        """Personen unter 14 Jahren sollten nicht erwerbstätig sein"""
        def check(row):
            if 'alter' in row and 'erwerbsstatus' in row:
                age = row['alter']
                employed = row['erwerbsstatus'] in ['vollzeit', 'teilzeit', 'angestellt', 'selbstständig']
                
                if age < 14 and employed:
                    return ValidationResult(
                        rule_name="age_employment_consistency",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Person mit {age} Jahren kann nicht erwerbstätig sein",
                        affected_fields=['alter', 'erwerbsstatus']
                    )
            return ValidationResult(
                rule_name="age_employment_consistency",
                passed=True,
                level=ValidationLevel.INFO,
                message="Alter-Erwerbsstatus konsistent",
                affected_fields=['alter', 'erwerbsstatus']
            )
        
        return PlausibilityRule(
            name="age_employment_consistency",
            description="Personen unter 14 Jahren sollten nicht erwerbstätig sein",
            check_function=check,
            level=ValidationLevel.ERROR,
            category="demographics"
        )
    
    @staticmethod
    def income_age_plausibility():
        """Einkommen sollte zu Alter passen"""
        def check(row):
            if 'alter' in row and 'einkommen' in row:
                age = row['alter']
                income = row['einkommen']
                
                # Sehr hohes Einkommen bei sehr jungen Personen unwahrscheinlich
                if age < 25 and income > 100000:
                    return ValidationResult(
                        rule_name="income_age_plausibility",
                        passed=False,
                        level=ValidationLevel.WARNING,
                        message=f"Sehr hohes Einkommen ({income}€) bei Person mit {age} Jahren ungewöhnlich",
                        affected_fields=['alter', 'einkommen']
                    )
                
                # Negatives Einkommen unrealistisch
                if income < 0:
                    return ValidationResult(
                        rule_name="income_age_plausibility",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Negatives Einkommen ({income}€) unrealistisch",
                        affected_fields=['einkommen']
                    )
            
            return ValidationResult(
                rule_name="income_age_plausibility",
                passed=True,
                level=ValidationLevel.INFO,
                message="Einkommen-Alter Verhältnis plausibel",
                affected_fields=['alter', 'einkommen']
            )
        
        return PlausibilityRule(
            name="income_age_plausibility",
            description="Einkommen sollte zum Alter passen",
            check_function=check,
            level=ValidationLevel.WARNING,
            category="economics"
        )
    
    @staticmethod
    def alcohol_abstinence_contradiction():
        """Alkoholverzicht vs. Alkoholkonsum Widerspruch"""
        def check(row):
            if 'alkoholverzicht_wichtig' in row and 'praef_bier_alkohol' in row:
                abstinence = row['alkoholverzicht_wichtig']
                alcohol_pref = row['praef_bier_alkohol']
                
                # Starker Alkoholverzicht (5) mit hoher Alkoholpräferenz (4-5) widerspricht sich
                if abstinence == 5 and alcohol_pref >= 4:
                    return ValidationResult(
                        rule_name="alcohol_abstinence_contradiction",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Widerspruch: Alkoholverzicht sehr wichtig ({abstinence}) aber hohe Alkoholpräferenz ({alcohol_pref})",
                        affected_fields=['alkoholverzicht_wichtig', 'praef_bier_alkohol']
                    )
            
            return ValidationResult(
                rule_name="alcohol_abstinence_contradiction",
                passed=True,
                level=ValidationLevel.INFO,
                message="Alkoholverzicht-Präferenz konsistent",
                affected_fields=['alkoholverzicht_wichtig', 'praef_bier_alkohol']
            )
        
        return PlausibilityRule(
            name="alcohol_abstinence_contradiction",
            description="Alkoholverzicht sollte nicht mit hoher Alkoholpräferenz einhergehen",
            check_function=check,
            level=ValidationLevel.ERROR,
            category="lifestyle"
        )
    
    @staticmethod
    def frequency_interest_consistency():
        """Konsumhäufigkeit sollte zu Interesse passen"""
        def check(row):
            if 'konsumhaeufigkeit' in row and 'involvement_interesse' in row:
                frequency = row['konsumhaeufigkeit']
                interest = row['involvement_interesse']
                
                # Kein Interesse (1) aber häufiger Konsum
                high_frequency = frequency in ['täglich', 'mehrmals pro Woche']
                no_interest = interest == 1
                
                if no_interest and high_frequency:
                    return ValidationResult(
                        rule_name="frequency_interest_consistency",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Widerspruch: Kein Interesse ({interest}) aber häufiger Konsum ({frequency})",
                        affected_fields=['konsumhaeufigkeit', 'involvement_interesse']
                    )
            
            return ValidationResult(
                rule_name="frequency_interest_consistency",
                passed=True,
                level=ValidationLevel.INFO,
                message="Konsumhäufigkeit-Interesse konsistent",
                affected_fields=['konsumhaeufigkeit', 'involvement_interesse']
            )
        
        return PlausibilityRule(
            name="frequency_interest_consistency",
            description="Häufiger Konsum sollte mit Interesse einhergehen",
            check_function=check,
            level=ValidationLevel.ERROR,
            category="behavior"
        )
    
    @staticmethod
    def health_alcohol_contradiction():
        """Gesundheitsbewusstsein vs. Alkoholkonsum"""
        def check(row):
            if 'gesundheit_wichtig' in row and 'alkoholverzicht_wichtig' in row and 'involvement_vorteilhaft' in row:
                health_important = row['gesundheit_wichtig']
                abstinence_important = row['alkoholverzicht_wichtig']
                beer_beneficial = row['involvement_vorteilhaft']
                
                # Gesundheit sehr wichtig (5) UND Alkoholverzicht sehr wichtig (5) 
                # aber Bier wird als sehr vorteilhaft (>=4) gesehen
                if health_important == 5 and abstinence_important == 5 and beer_beneficial >= 4:
                    return ValidationResult(
                        rule_name="health_alcohol_contradiction",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Widerspruch: Gesundheit & Alkoholverzicht sehr wichtig aber Bier als vorteilhaft ({beer_beneficial}) bewertet",
                        affected_fields=['gesundheit_wichtig', 'alkoholverzicht_wichtig', 'involvement_vorteilhaft']
                    )
            
            return ValidationResult(
                rule_name="health_alcohol_contradiction",
                passed=True,
                level=ValidationLevel.INFO,
                message="Gesundheitsbewusstsein-Alkohol Einstellung konsistent",
                affected_fields=['gesundheit_wichtig', 'alkoholverzicht_wichtig', 'involvement_vorteilhaft']
            )
        
        return PlausibilityRule(
            name="health_alcohol_contradiction",
            description="Starkes Gesundheitsbewusstsein sollte nicht mit positiver Alkoholbewertung einhergehen",
            check_function=check,
            level=ValidationLevel.ERROR,
            category="lifestyle"
        )
    
    @staticmethod
    def sustainability_payment_consistency():
        """Nachhaltigkeitswerte vs. Zahlungsbereitschaft"""
        def check(row):
            if 'einfluss_nachhaltigkeit' in row and 'bio_mehrpreis_bereit' in row:
                sustainability_influence = row['einfluss_nachhaltigkeit']
                willing_to_pay_more = row['bio_mehrpreis_bereit']
                
                # Nachhaltigkeit sehr wichtig (5) aber keine Bereitschaft Mehrpreis zu zahlen (<=2)
                if sustainability_influence == 5 and willing_to_pay_more <= 2:
                    return ValidationResult(
                        rule_name="sustainability_payment_consistency",
                        passed=False,
                        level=ValidationLevel.WARNING,
                        message=f"Widerspruch: Nachhaltigkeit sehr wichtig ({sustainability_influence}) aber geringe Mehrpreis-Bereitschaft ({willing_to_pay_more})",
                        affected_fields=['einfluss_nachhaltigkeit', 'bio_mehrpreis_bereit']
                    )
            
            return ValidationResult(
                rule_name="sustainability_payment_consistency",
                passed=True,
                level=ValidationLevel.INFO,
                message="Nachhaltigkeitswerte-Zahlungsbereitschaft konsistent",
                affected_fields=['einfluss_nachhaltigkeit', 'bio_mehrpreis_bereit']
            )
        
        return PlausibilityRule(
            name="sustainability_payment_consistency",
            description="Hohe Nachhaltigkeitswerte sollten mit Bereitschaft für Mehrpreis einhergehen",
            check_function=check,
            level=ValidationLevel.WARNING,
            category="values"
        )
    
    @staticmethod
    def data_range_validity():
        """Überprüft ob Werte in plausiblen Bereichen liegen"""
        def check(row):
            results = []
            
            # Alter sollte zwischen 16 und 99 sein (für Umfrage über Alkohol)
            if 'alter' in row:
                age = row['alter']
                if age < 16 or age > 99:
                    results.append(ValidationResult(
                        rule_name="data_range_validity",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Alter {age} außerhalb plausiblem Bereich (16-99)",
                        affected_fields=['alter']
                    ))
            
            # Likert-Skalen sollten zwischen 1 und 5 sein
            likert_fields = ['praef_bier_alkohol', 'alkoholverzicht_wichtig', 'gesundheit_wichtig', 
                           'involvement_interesse', 'involvement_rolle', 'involvement_vorteilhaft',
                           'einfluss_nachhaltigkeit', 'bio_mehrpreis_bereit']
            
            for field in likert_fields:
                if field in row:
                    value = row[field]
                    if value < 1 or value > 5:
                        results.append(ValidationResult(
                            rule_name="data_range_validity",
                            passed=False,
                            level=ValidationLevel.ERROR,
                            message=f"{field} Wert {value} außerhalb Skala (1-5)",
                            affected_fields=[field]
                        ))
            
            # Wenn keine Probleme gefunden
            if not results:
                return ValidationResult(
                    rule_name="data_range_validity",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message="Alle Werte in plausiblen Bereichen",
                    affected_fields=[]
                )
            
            return results[0]  # Gib ersten Fehler zurück
        
        return PlausibilityRule(
            name="data_range_validity",
            description="Überprüft ob alle Werte in plausiblen Bereichen liegen",
            check_function=check,
            level=ValidationLevel.ERROR,
            category="validity"
        )

def create_beer_survey_checker() -> PlausibilityChecker:
    """Erstellt vorkonfigurierten Checker für Bier-Umfrage Daten"""
    checker = PlausibilityChecker()
    
    # Füge alle Standard-Regeln hinzu
    checker.add_rule(StandardPlausibilityRules.age_employment_consistency())
    checker.add_rule(StandardPlausibilityRules.income_age_plausibility())
    checker.add_rule(StandardPlausibilityRules.alcohol_abstinence_contradiction())
    checker.add_rule(StandardPlausibilityRules.frequency_interest_consistency())
    checker.add_rule(StandardPlausibilityRules.health_alcohol_contradiction())
    checker.add_rule(StandardPlausibilityRules.sustainability_payment_consistency())
    checker.add_rule(StandardPlausibilityRules.data_range_validity())
    
    return checker

# Beispiel für die Verwendung
if __name__ == "__main__":
    # Erstelle Beispiel-Daten mit bewussten Inkonsistenzen
    np.random.seed(42)
    
    # Simuliere problematische Daten
    data = {
        'alter': [12, 25, 30, 22, 45],  # 12-Jähriger problematisch
        'einkommen': [0, 35000, 45000, 150000, 55000],  # 150k bei 22-Jährigem fragwürdig
        'erwerbsstatus': ['vollzeit', 'teilzeit', 'vollzeit', 'selbstständig', 'vollzeit'],  # 12-Jähriger vollzeit problematisch
        'konsumhaeufigkeit': ['täglich', 'wöchentlich', 'monatlich', 'täglich', 'wöchentlich'],
        'praef_bier_alkohol': [5, 3, 2, 5, 4],  # Hohe Präferenz
        'alkoholverzicht_wichtig': [5, 2, 4, 1, 3],  # 5 bei erstem = Widerspruch
        'gesundheit_wichtig': [5, 3, 4, 2, 5],
        'involvement_interesse': [1, 3, 2, 4, 3],  # 1 bei täglich = Widerspruch
        'involvement_rolle': [2, 3, 2, 4, 3],
        'involvement_vorteilhaft': [5, 2, 3, 3, 2],  # 5 bei gesundheit_wichtig=5 + alkoholverzicht=5
        'einfluss_nachhaltigkeit': [5, 2, 3, 4, 5],
        'bio_mehrpreis_bereit': [1, 3, 3, 4, 4]  # 1 bei nachhaltigkeit=5 = Widerspruch
    }
    
    df = pd.DataFrame(data)
    
    print("Beispiel-Daten:")
    print(df.head())
    
    # Erstelle und führe Plausibilitätsprüfung durch
    checker = create_beer_survey_checker()
    summary = checker.validate_dataset(df)
    
    # Zeige Ergebnisse
    checker.print_summary(summary)
    
    # Zeige detaillierte Fehler
    print("\n🔍 DETAILLIERTE FEHLER:")
    error_results = [r for r in summary['detailed_results'] 
                    if not r.passed and r.level == ValidationLevel.ERROR]
    
    for result in error_results[:5]:  # Zeige ersten 5 Fehler
        print(f"   Row {result.row_index}: {result.message}")