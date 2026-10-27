"""Inventario de fuentes públicas para CelsiaBot.

Las URL deben comprobarse antes de utilizarlas en la extracción.
Priorizar fuentes oficiales de Celsia.
"""

SOURCES = [
    {
        "title": "Sitio web oficial de Celsia",
        "url": "https://www.celsia.com/es/",
        "category": "corporativo",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Sobre nosotros",
        "url": "https://www.celsia.com/es/ley-de-transparencia-y-acceso-a-la-informacion/sobre-nosotros/",
        "category": "corporativo",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Gobierno corporativo",
        "url": "https://www.celsia.com/es/quienes-somos/gobierno-corporativo/",
        "category": "gobierno_corporativo",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Sostenibilidad",
        "url": "https://www.celsia.com/es/sostenibilidad/",
        "category": "sostenibilidad",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Informes de sostenibilidad",
        "url": "https://www.celsia.com/es/sostenibilidad/reportes/",
        "category": "informes",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Sala de prensa",
        "url": "https://www.celsia.com/es/sala-de-prensa/",
        "category": "noticias",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Granjas solares",
        "url": "https://www.celsia.com/es/granjas-solares/",
        "category": "energia_solar",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Oportunidades laborales",
        "url": "https://www.celsia.com/es/se-parte-de-nuestro-equipo/",
        "category": "empleo",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Canales de contacto",
        "url": "https://www.celsia.com/es/encuentranos/",
        "category": "atencion",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
    {
        "title": "Información para inversionistas",
        "url": "https://www.celsia.com/es/inversionistas/celsia/",
        "category": "inversionistas",
        "source_type": "web",
        "official": True,
        "requires_verification": True,
    },
]


def get_sources(category=None):
    """Devuelve todas las fuentes o las de una categoría concreta."""
    if category is None:
        return SOURCES.copy()

    return [
        source
        for source in SOURCES
        if source["category"] == category
    ]
