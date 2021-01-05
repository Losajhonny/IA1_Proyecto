import { Component } from '@angular/core';
import { Conexion } from "./Conexion";

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})

export class AppComponent {
    deptos = [];
    municipios = [];

    data = {
        genero: "1",
        edad: 20,
        anio: 2020,
        depto: "-1",
        municipio: "-1"
    };

    constructor() {
        this.getDeptos();
    }

    async getDeptos() {
        this.deptos = (await Conexion.getInstance().GET("deptos")).data;
    }

    async loadMuni(codigo) {
        if (codigo == -1)
            return;
        this.municipios = (await Conexion.getInstance().POST('munic', { codigo: codigo })).data
    }

    async consultar() {
        const res = (await Conexion.getInstance().POST('consultar', { data: this.data })).data
        const aux = "Probabilidad que el estudiante se cambie de carrera es de: " + res
        alert(aux)
    }
}
