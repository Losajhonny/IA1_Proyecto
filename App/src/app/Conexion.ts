export class Conexion {

    // direccion del servidor
    private server: string = "http://localhost:5000/";

    // singleton conexion
    private static CONEXION: Conexion;

    public static getInstance(): Conexion {
       if (this.CONEXION === null || this.CONEXION === undefined) {
          this.CONEXION = new Conexion();
       }
       return this.CONEXION;
    }

    async POST(url: string, request: any) {
       url = this.server + url;
       console.log(url);

       const option = {
          method: 'POST',
          headers: { 'Accept': 'application/json', 'Content-Type': 'application/json' },
          body: JSON.stringify(request)
       }

       let res: any = null;

       await fetch(url, option)
          .then((res) => {
             return res.json();
          })
          .then((data) => {
             res = data;
          })
          .catch((ex) => {
             console.log(ex);
             res = null;
          });

       return res;
    }

    async GET(url: string) {
       url = this.server + url;
       console.log(url);

       const option = {
          method: 'GET',
          headers: { 'Accept': 'application/json', 'Content-Type': 'application/json' }
       }

       let res: any = null;

       await fetch(url, option)
          .then((res) => {
             return res.json();
          })
          .then((data) => {
             res = data;
          })
          .catch((ex) => {
             console.log(ex);
             res = null;
          });

       return res;
    }
 }
